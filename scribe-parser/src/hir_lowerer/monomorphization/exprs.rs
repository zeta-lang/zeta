use ir::{
    hir::{
        self, HirExpr, HirFieldInit, HirFunc, HirGeneric, HirMatchArm, HirModuleAccess, HirParam,
        HirStmt, HirStruct, HirType, InterpolationPart, IntrinsicKind, StrId,
    },
    ir_hasher::{FxHashMap, HashMap},
};

use crate::hir_lowerer::{
    HirLowerer,
    monomorphization::{
        Monomorphizer,
        assertions::{contains_unresolved_generic, peel_to_struct, peel_to_struct_owned},
        instantiate_struct_for_types,
        struct_instantiation::instantiate_enum_for_types,
        substitute_type,
    },
};

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    pub fn monomorphize_expr<'subs>(
        &self,
        expr: &HirExpr<'a, 'bump>,
        subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirExpr<'a, 'bump> {
        match expr {
            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
            } => {
                let new_args: Vec<HirExpr> = args
                    .iter()
                    .map(|a| self.monomorphize_expr(a, subs))
                    .collect();
                let args_slice = self.bump.alloc_slice(&new_args);

                if let Some(targs) = type_args {
                    let resolved_targs: Vec<HirType> = targs
                        .iter()
                        .map(|t| substitute_type(t, subs, &self.bump))
                        .collect();

                    let base_variants = self.ctx.enums.borrow().get(enum_name).map(|e| e.variants);

                    if let Some(_) = base_variants {
                        if let Some(new_enum) = instantiate_enum_for_types(
                            self.ctx,
                            &self.instantiated_enums,
                            &self.instantiated_enum_origins,
                            *enum_name,
                            &resolved_targs,
                            &&self.bump,
                        ) {
                            return HirExpr::EnumInit {
                                enum_name: new_enum.name,
                                variant: *variant,
                                args: args_slice,
                                type_args: None,
                                span: *span,
                            };
                        }
                    }
                }

                HirExpr::EnumInit {
                    enum_name: *enum_name,
                    variant: *variant,
                    args: args_slice,
                    type_args: *type_args,
                    span: *span,
                }
            }
            HirExpr::Cast {
                expr,
                target_type,
                span,
            } => HirExpr::Cast {
                expr: self
                    .bump
                    .alloc_value_immutable(self.monomorphize_expr(expr, subs)),
                target_type: {
                    let substituted = self.substitute_type_with_self(target_type, subs);
                    self.instantiate_type_recursively(substituted, *span)
                },
                span: *span,
            },
            HirExpr::Index {
                object,
                index,
                span,
            } => {
                let new_object = self.monomorphize_expr(object, subs);
                let new_index = self.monomorphize_expr(index, subs);
                HirExpr::Index {
                    object: self.bump.alloc_value_immutable(new_object),
                    index: self.bump.alloc_value_immutable(new_index),
                    span: *span,
                }
            }
            HirExpr::Intrinsic {
                kind,
                type_args,
                args,
                span,
            } => {
                let new_type_args: Vec<HirType> = type_args
                    .iter()
                    .map(|t| self.substitute_type_with_self(t, subs))
                    .collect();

                let new_args: Vec<HirExpr> = args
                    .iter()
                    .map(|a| self.monomorphize_expr(a, subs))
                    .collect();
                HirExpr::Intrinsic {
                    kind: *kind,
                    type_args: self.bump.alloc_slice(&new_type_args),
                    args: self.bump.alloc_slice(&new_args),
                    span: *span,
                }
            }
            HirExpr::Call {
                callee,
                args,
                type_args,
                span,
            } => {
                if let (HirExpr::Ident(func_name, ident_span), None) = (&**callee, type_args) {
                    let maybe_func = self.functions.borrow().get(func_name).cloned();
                    if let Some(func) = maybe_func {
                        if let (Some(type_params), Some(params)) = (func.generics, func.params) {
                            if !type_params.is_empty() {
                                let declared: Vec<HirType> = params
                                    .iter()
                                    .filter_map(|p| match p {
                                        HirParam::Normal { param_type, .. } => Some(*param_type),
                                        _ => None,
                                    })
                                    .collect();

                                let mut inner_subs: FxHashMap<StrId, HirType> =
                                    FxHashMap::default();

                                // A hoisted closure literal is `StructInit { name: __closure_env_N }`;
                                // it binds the `F: func(..)` generic to the env struct.
                                for (declared_ty, arg) in declared.iter().zip(args.iter()) {
                                    if let (
                                        HirType::Generic(g),
                                        HirExpr::StructInit {
                                            name,
                                            type_args: None,
                                            ..
                                        },
                                    ) = (declared_ty, arg)
                                    {
                                        if let HirExpr::Ident(env_name, _) = &**name {
                                            let is_closure_param = type_params
                                                .iter()
                                                .find(|p| p.name == *g)
                                                .map_or(false, |p| {
                                                    p.constraints.iter().any(|c| {
                                                        matches!(c, HirType::Lambda { .. })
                                                    })
                                                });

                                            if self.env_structs.contains_key(env_name)
                                                && is_closure_param
                                            {
                                                inner_subs
                                                    .insert(*g, self.closure_env_type(*env_name));
                                            }
                                        }
                                    }
                                }

                                // Everything else (e.g. `T` from a non-closure argument).
                                self.infer_missing_generics(
                                    type_params,
                                    &declared,
                                    args,
                                    subs,
                                    &mut inner_subs,
                                );

                                // Infer chained generics that appear only in closure
                                // constraints, e.g. `R` in `F: func(): R`.
                                self.infer_generics_from_closure_constraints(
                                    type_params,
                                    &inner_subs.clone(),
                                    &mut inner_subs,
                                );

                                if inner_subs.len() == type_params.len() {
                                    let new_args: Vec<HirExpr> = args
                                        .iter()
                                        .map(|a| self.monomorphize_expr(a, subs))
                                        .collect();
                                    if let Some(new_name) =
                                        self.monomorphize_function(&func, &inner_subs)
                                    {
                                        return HirExpr::Call {
                                            callee: self.bump.alloc_value_immutable(
                                                HirExpr::Ident(new_name, *ident_span),
                                            ),
                                            args: self.bump.alloc_slice(&new_args),
                                            type_args: None,
                                            span: *span,
                                        };
                                    }
                                }
                            }
                        }
                    }
                }

                // Handle calls to generic free functions via a module path,
                // e.g. `threads.spawn(func() { ... })`.
                // The HIR lowerer leaves such callees as ModuleAccess when the
                // function is not a re-exported facade, so the monomorphizer
                // must resolve and instantiate them here.
                if let (HirExpr::ModuleAccess(acc), None) = (&**callee, type_args) {
                    if let Some(func_name) = self.resolve_module_access_function(acc) {
                        let maybe_func = self.functions.borrow().get(&func_name).cloned();
                        if let Some(func) = maybe_func {
                            if let (Some(type_params), Some(params)) = (func.generics, func.params)
                            {
                                if !type_params.is_empty() {
                                    let declared: Vec<HirType> = params
                                        .iter()
                                        .filter_map(|p| match p {
                                            HirParam::Normal { param_type, .. } => {
                                                Some(*param_type)
                                            }
                                            _ => None,
                                        })
                                        .collect();

                                    let mut inner_subs: FxHashMap<StrId, HirType> =
                                        FxHashMap::default();

                                    // Bind closure-env generics.
                                    for (declared_ty, arg) in declared.iter().zip(args.iter()) {
                                        if let (
                                            HirType::Generic(g),
                                            HirExpr::StructInit {
                                                name,
                                                type_args: None,
                                                ..
                                            },
                                        ) = (declared_ty, arg)
                                        {
                                            if let HirExpr::Ident(env_name, _) = &**name {
                                                let is_closure_param = type_params
                                                    .iter()
                                                    .find(|p| p.name == *g)
                                                    .map_or(false, |p| {
                                                        p.constraints.iter().any(|c| {
                                                            matches!(c, HirType::Lambda { .. })
                                                        })
                                                    });

                                                if self.env_structs.contains_key(env_name)
                                                    && is_closure_param
                                                {
                                                    inner_subs.insert(
                                                        *g,
                                                        self.closure_env_type(*env_name),
                                                    );
                                                }
                                            }
                                        }
                                    }

                                    self.infer_missing_generics(
                                        type_params,
                                        &declared,
                                        args,
                                        subs,
                                        &mut inner_subs,
                                    );

                                    // Infer chained generics that appear only in closure
                                    // constraints, e.g. `R` in `F: func(): R`.
                                    self.infer_generics_from_closure_constraints(
                                        type_params,
                                        &inner_subs.clone(),
                                        &mut inner_subs,
                                    );

                                    if inner_subs.len() == type_params.len() {
                                        let new_args: Vec<HirExpr> = args
                                            .iter()
                                            .map(|a| self.monomorphize_expr(a, subs))
                                            .collect();
                                        if let Some(new_name) =
                                            self.monomorphize_function(&func, &inner_subs)
                                        {
                                            return HirExpr::Call {
                                                callee: self.bump.alloc_value_immutable(
                                                    HirExpr::Ident(new_name, acc.span),
                                                ),
                                                args: self.bump.alloc_slice(&new_args),
                                                type_args: None,
                                                span: *span,
                                            };
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                if let HirExpr::FieldAccess {
                    object,
                    field,
                    span: fa_span,
                } = &**callee
                {
                    let new_object = self.monomorphize_expr(object, subs);

                    if let Some(resolved) = self.resolve_own_generic_method_call(
                        &new_object,
                        *field,
                        *type_args,
                        args,
                        subs,
                    ) {
                        return resolved;
                    }

                    if let Some(concrete_recv_ty) = self.concrete_type_of(&new_object) {
                        let concrete_recv_ty = peel_to_struct_owned(concrete_recv_ty);
                        if matches!(
                            concrete_recv_ty,
                            HirType::Struct { .. } | HirType::Enum { .. }
                        ) {
                            if let Some(concrete_method_name) = self.resolve_method_for_type(
                                &concrete_recv_ty,
                                *field,
                                *type_args,
                                subs,
                            ) {
                                let method_func =
                                    self.functions.borrow().get(&concrete_method_name).cloned();
                                let is_instance_method = method_func.as_ref().map_or(false, |f| {
                                    f.params.as_ref().map_or(false, |p| {
                                        matches!(p.first(), Some(HirParam::This { .. }))
                                    })
                                });

                                let mut new_args: Vec<HirExpr> = Vec::with_capacity(args.len() + 1);
                                if is_instance_method {
                                    new_args.push(new_object.clone());
                                }
                                new_args
                                    .extend(args.iter().map(|a| self.monomorphize_expr(a, subs)));
                                let args_slice = self.bump.alloc_slice(&new_args);
                                let new_callee = HirExpr::Ident(concrete_method_name, *fa_span);
                                return HirExpr::Call {
                                    callee: self.bump.alloc_value_immutable(new_callee),
                                    args: args_slice,
                                    type_args: None,
                                    span: *span,
                                };
                            }
                        }
                    }

                    let new_args: Vec<HirExpr> = args
                        .iter()
                        .map(|a| self.monomorphize_expr(a, subs))
                        .collect();
                    let args_slice = self.bump.alloc_slice(&new_args);
                    let new_callee = HirExpr::FieldAccess {
                        object: self.bump.alloc_value_immutable(new_object),
                        field: *field,
                        span: *fa_span,
                    };
                    let new_type_args = type_args.map(|targs| {
                        let subd: Vec<HirType> = targs
                            .iter()
                            .map(|t| substitute_type(t, subs, &self.bump))
                            .collect();
                        &*self.bump.alloc_slice(&subd)
                    });
                    return HirExpr::Call {
                        callee: self.bump.alloc_value_immutable(new_callee),
                        args: args_slice,
                        type_args: new_type_args,
                        span: *span,
                    };
                }

                // Handle static method calls on generic types via ModuleAccess,
                // e.g. `ArrayList.with_capacity<u8, A>(...)`.
                if let HirExpr::ModuleAccess(acc) = &**callee {
                    if let Some(targs) = type_args {
                        let (&struct_name, module_path) = acc.path.split_last().unwrap_or_else(|| {
                        panic!(
                            "monomorphize_expr: empty module path in static call `.{}<...>` at {span}",
                            self.context.resolve_string(&acc.member)
                        )
                    });
                        let target_key = if self.ctx.structs.borrow().contains_key(&struct_name) {
                            struct_name
                        } else {
                            self.ctx
                                .resolve_type_path_name(module_path, struct_name, acc.span)
                        };

                        let mut resolved_targs: Vec<HirType> = targs
                            .iter()
                            .map(|t| substitute_type(t, subs, &self.bump))
                            .collect();

                        if let Some(ty_struct) = self.ctx.structs.borrow().get(&target_key) {
                            if let Some(declared) = ty_struct.generics {
                                HirLowerer::fill_default_type_args(
                                    declared,
                                    &mut resolved_targs,
                                    &&self.bump,
                                );
                            }
                        }

                        let struct_ty = HirType::Struct {
                            name: target_key,
                            field_types: &[],
                            type_args: self.bump.alloc_slice(&resolved_targs),
                        };

                        let concrete_method_name = self
                        .resolve_method_for_type(&struct_ty, acc.member, None, subs)
                        .unwrap_or_else(|| {
                            panic!(
                                "monomorphize_expr: could not resolve `{}.{}<...>` at {span}; struct key \
                                 `{}` with type args {:?} has no such method (or the struct/method \
                                 couldn't be instantiated).",
                                struct_name,
                                acc.member,
                                target_key,
                                resolved_targs,
                            )
                        });

                        let new_args: Vec<HirExpr> = args
                            .iter()
                            .map(|a| self.monomorphize_expr(a, subs))
                            .collect();
                        let args_slice = self.bump.alloc_slice(&new_args);
                        let new_callee = HirExpr::Ident(concrete_method_name, acc.span);
                        return HirExpr::Call {
                            callee: self.bump.alloc_value_immutable(new_callee),
                            args: args_slice,
                            type_args: None,
                            span: *span,
                        };
                    }
                }

                let new_callee = self.monomorphize_expr(callee, subs);
                let new_args: Vec<HirExpr> = args
                    .iter()
                    .map(|a| self.monomorphize_expr(a, subs))
                    .collect();
                let args_slice = self.bump.alloc_slice(&new_args);
                HirExpr::Call {
                    callee: self.bump.alloc_value_immutable(new_callee),
                    args: args_slice,
                    type_args: *type_args,
                    span: *span,
                }
            }
            HirExpr::InterfaceCall {
                callee,
                interface,
                args,
                span,
            } => {
                let new_callee = self.monomorphize_expr(callee, subs);
                let new_args: Vec<HirExpr> = args
                    .iter()
                    .map(|a| self.monomorphize_expr(a, subs))
                    .collect();
                let args_slice = self.bump.alloc_slice(&new_args);
                HirExpr::InterfaceCall {
                    callee: self.bump.alloc_value_immutable(new_callee),
                    interface: *interface,
                    args: args_slice,
                    span: *span,
                }
            }
            HirExpr::StructInit {
                name,
                args,
                type_args,
                span,
            } => {
                let new_args: Vec<HirFieldInit<'a, 'bump>> = args
                    .iter()
                    .map(|a| HirFieldInit {
                        name: a.name,
                        name_span: a.name_span,
                        value: self.monomorphize_expr(&a.value, subs),
                    })
                    .collect();
                let args_slice = self.bump.alloc_slice(&new_args);

                if let (HirExpr::Ident(struct_name, ident_span), Some(targs)) = (&**name, type_args)
                {
                    let resolved_targs: Vec<HirType> = targs
                        .iter()
                        .map(|t| substitute_type(t, subs, &self.bump))
                        .collect();

                    if let Some(new_struct) = instantiate_struct_for_types(
                        self.ctx,
                        &self.instantiated_structs,
                        &self.instantiated_struct_origins,
                        &self.instantiated_enums,
                        &self.instantiated_enum_origins,
                        *struct_name,
                        &resolved_targs,
                        &&self.bump,
                    ) {
                        let new_name_expr = HirExpr::Ident(new_struct.name, *ident_span);
                        return HirExpr::StructInit {
                            name: self.bump.alloc_value_immutable(new_name_expr),
                            args: args_slice,
                            type_args: None,
                            span: *span,
                        };
                    }
                }

                let new_name = self.monomorphize_expr(name, subs);
                HirExpr::StructInit {
                    name: self.bump.alloc_value_immutable(new_name),
                    args: args_slice,
                    type_args: *type_args,
                    span: *span,
                }
            }
            HirExpr::FieldAccess {
                object,
                field,
                span,
            } => {
                let new_object = self.monomorphize_expr(object, subs);
                #[cfg(debug_assertions)]
                if let Some(recv) = self.concrete_type_of(&new_object) {
                    if let HirType::Struct { name: rn, .. } = peel_to_struct_owned(recv) {
                        let origin = self
                            .instantiated_struct_origins
                            .borrow()
                            .get(&rn)
                            .map(|(o, _)| *o)
                            .unwrap_or(rn);
                        let is_generic_method = self
                            .ctx
                            .struct_methods
                            .borrow()
                            .get(&origin)
                            .and_then(|m| m.get(field))
                            .and_then(|n| self.functions.borrow().get(n).cloned())
                            .map_or(false, |f| f.generics.map_or(false, |g| !g.is_empty()));
                        if is_generic_method {
                            panic!(
                                "monomorphize_expr: generic method `{}` on `{}` left uninstantiated at {span}",
                                field, origin
                            );
                        }
                    }
                }
                HirExpr::FieldAccess {
                    object: self.bump.alloc_value_immutable(new_object),
                    field: *field,
                    span: *span,
                }
            }
            HirExpr::Get {
                object,
                field,
                span,
            } => {
                let new_object = self.monomorphize_expr(object, subs);
                HirExpr::Get {
                    object: self.bump.alloc_value_immutable(new_object),
                    field: *field,
                    span: *span,
                }
            }
            HirExpr::Binary {
                left,
                op,
                right,
                span,
            } => {
                let new_left = self.monomorphize_expr(left, subs);
                let new_right = self.monomorphize_expr(right, subs);
                HirExpr::Binary {
                    left: self.bump.alloc_value_immutable(new_left),
                    op: *op,
                    right: self.bump.alloc_value_immutable(new_right),
                    span: *span,
                }
            }
            HirExpr::Assignment {
                target,
                op,
                value,
                span,
            } => {
                let new_target = self.monomorphize_expr(target, subs);
                let new_value = match self.concrete_type_of(target) {
                    Some(expected_ty) => {
                        self.monomorphize_expr_with_expected_type(value, &expected_ty, subs)
                    }
                    None => self.monomorphize_expr(value, subs),
                };
                HirExpr::Assignment {
                    target: self.bump.alloc_value_immutable(new_target),
                    op: *op,
                    value: self.bump.alloc_value_immutable(new_value),
                    span: *span,
                }
            }
            HirExpr::ExprList { list, span } => {
                let new_list: Vec<HirExpr> = list
                    .iter()
                    .map(|e| self.monomorphize_expr(e, subs))
                    .collect();
                let list_slice = self.bump.alloc_slice(&new_list);
                HirExpr::ExprList {
                    list: list_slice,
                    span: *span,
                }
            }
            HirExpr::Comparison {
                left,
                op,
                right,
                span,
            } => {
                let new_left = self.monomorphize_expr(left, subs);
                let new_right = self.monomorphize_expr(right, subs);
                HirExpr::Comparison {
                    left: self.bump.alloc_value_immutable(new_left),
                    op: *op,
                    right: self.bump.alloc_value_immutable(new_right),
                    span: *span,
                }
            }
            HirExpr::Deref { expr, span } => {
                let new_expr = self.monomorphize_expr(expr, subs);
                HirExpr::Deref {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    span: *span,
                }
            }
            HirExpr::Ref {
                expr,
                ref_kind,
                span,
            } => {
                let new_expr = self.monomorphize_expr(expr, subs);
                HirExpr::Ref {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    ref_kind: *ref_kind,
                    span: *span,
                }
            }
            HirExpr::ArrayLiteral { elements, span } => {
                let new_elems: Vec<HirExpr> = elements
                    .iter()
                    .map(|e| self.monomorphize_expr(e, subs))
                    .collect();
                let elems_slice = self.bump.alloc_slice(&new_elems);
                HirExpr::ArrayLiteral {
                    elements: elems_slice,
                    span: *span,
                }
            }
            HirExpr::Match { expr, arms, span } => {
                let new_expr = self.monomorphize_expr(expr, subs);
                let new_arms: Vec<HirMatchArm> = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.map(|g| {
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(g, subs))
                        }),
                        body: self
                            .bump
                            .alloc_value_immutable(self.monomorphize_stmt(arm.body, subs)),
                    })
                    .collect();
                HirExpr::Match {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                    span: *span,
                }
            }
            HirExpr::Block {
                body,
                is_unsafe,
                span,
            } => {
                let new_body: Vec<HirStmt> = body
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, subs))
                    .collect();
                HirExpr::Block {
                    body: self.bump.alloc_slice(&new_body),
                    is_unsafe: *is_unsafe,
                    span: *span,
                }
            }
            HirExpr::If { if_stmt, span } => {
                let new_stmt = self.monomorphize_stmt(if_stmt, subs);
                HirExpr::If {
                    if_stmt: self.bump.alloc_value_immutable(new_stmt),
                    span: *span,
                }
            }
            HirExpr::Slice {
                object,
                start,
                end,
                inclusive,
                span,
            } => {
                let new_object = self.monomorphize_expr(object, subs);
                let new_start = self.monomorphize_expr(start, subs);
                let new_end = self.monomorphize_expr(end, subs);
                HirExpr::Slice {
                    object: self.bump.alloc_value_immutable(new_object),
                    start: self.bump.alloc_value_immutable(new_start),
                    end: self.bump.alloc_value_immutable(new_end),
                    inclusive: *inclusive,
                    span: *span,
                }
            }
            HirExpr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                let new_start = self.monomorphize_expr(start, subs);
                let new_end = self.monomorphize_expr(end, subs);
                HirExpr::Range {
                    start: self.bump.alloc_value_immutable(new_start),
                    end: self.bump.alloc_value_immutable(new_end),
                    inclusive: *inclusive,
                    span: *span,
                }
            }
            HirExpr::Tuple(exprs, span) => {
                let new_exprs: Vec<HirExpr> = exprs
                    .iter()
                    .map(|e| self.monomorphize_expr(e, subs))
                    .collect();
                HirExpr::Tuple(self.bump.alloc_slice(&new_exprs), *span)
            }
            HirExpr::InterpolatedString(parts) => {
                let new_parts: Vec<InterpolationPart> = parts
                    .iter()
                    .map(|part| match part {
                        InterpolationPart::String(s) => InterpolationPart::String(*s),
                        InterpolationPart::Expr(e) => InterpolationPart::Expr(
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(e, subs)),
                        ),
                    })
                    .collect();
                HirExpr::InterpolatedString(self.bump.alloc_slice(&new_parts))
            }
            HirExpr::GenericIdent(name, type_args, span) => {
                let resolved_targs: Vec<HirType> = type_args
                    .iter()
                    .map(|t| substitute_type(t, subs, &self.bump))
                    .collect();

                let maybe_func = self.functions.borrow().get(name).cloned().or_else(|| {
                    let candidates: Vec<_> = self
                        .functions
                        .borrow()
                        .values()
                        .filter(|f| f.unmangled_name == *name && f.generics.is_some())
                        .cloned()
                        .collect();
                    match candidates.len() {
                        1 => Some(candidates.into_iter().next().unwrap()),
                        0 => None,
                        _ => panic!(
                            "monomorphize_expr: generic call `{}<...>` at {span} is ambiguous; {} \
                             distinct GENERIC templates named `{}` exist (declaring_module_idx: {:?}). \
                             This means two different modules genuinely declare a generic function with \
                             the same name, and GenericIdent needs its own module context recorded at \
                             lowering time to disambiguate between them.",
                            self.context.resolve_string(name),
                            candidates.len(),
                            self.context.resolve_string(name),
                            candidates.iter().map(|f| f.declaring_module_idx).collect::<Vec<_>>(),
                        ),
                    }
                });

                if let Some(func) = maybe_func {
                    if let Some(type_params) = func.generics {
                        if type_params.len() == resolved_targs.len() {
                            let mut inner_subs = FxHashMap::default();
                            for (param, arg) in type_params.iter().zip(resolved_targs.iter()) {
                                inner_subs.insert(param.name, *arg);
                            }
                            if let Some(new_fn) = self.monomorphize_function(&func, &inner_subs) {
                                return HirExpr::Ident(new_fn, *span);
                            }
                        }
                    }
                }

                panic!(
                    "monomorphize_expr: could not resolve generic call `{}<...>` at {span} to any \
                     function in `self.functions`, by mangled name or by (unmangled_name, \
                     declaring_module_idx={}); this function may not exist in this module, or may \
                     be registered under a different declaring_module_idx than expected.",
                    self.context.resolve_string(name),
                    self.ctx.module_idx,
                );
            }
            _ => expr.clone(),
        }
    }

    pub(crate) fn monomorphize_expr_with_expected_type<'subs>(
        &self,
        expr: &HirExpr<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirExpr<'a, 'bump> {
        match expr {
            HirExpr::Match {
                expr: scrutinee,
                arms,
                span,
            } => {
                let new_scrutinee = self.monomorphize_expr(scrutinee, subs);
                let new_arms: Vec<HirMatchArm> = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.map(|g| {
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(g, subs))
                        }),
                        body: self.bump.alloc_value_immutable(
                            self.monomorphize_stmt_with_expected_type(arm.body, expected_ty, subs),
                        ),
                    })
                    .collect();
                HirExpr::Match {
                    expr: self.bump.alloc_value_immutable(new_scrutinee),
                    arms: self.bump.alloc_slice(&new_arms),
                    span: *span,
                }
            }
            HirExpr::Block {
                body,
                is_unsafe,
                span,
            } => {
                let Some((last, rest)) = body.split_last() else {
                    return HirExpr::Block {
                        body: self.bump.alloc_slice(&[]),
                        is_unsafe: *is_unsafe,
                        span: *span,
                    };
                };
                let mut new_body: Vec<HirStmt> = rest
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, subs))
                    .collect();
                new_body.push(self.monomorphize_stmt_with_expected_type(last, expected_ty, subs));
                HirExpr::Block {
                    body: self.bump.alloc_slice(&new_body),
                    is_unsafe: *is_unsafe,
                    span: *span,
                }
            }
            HirExpr::If { if_stmt, span } => {
                let new_stmt =
                    self.monomorphize_stmt_with_expected_type(if_stmt, expected_ty, subs);
                HirExpr::If {
                    if_stmt: self.bump.alloc_value_immutable(new_stmt),
                    span: *span,
                }
            }
            _ => self
                .try_monomorphize_enum_init_with_expected_type(expr, expected_ty, subs)
                .or_else(|| {
                    self.try_monomorphize_struct_init_with_expected_type(expr, expected_ty, subs)
                })
                .unwrap_or_else(|| self.monomorphize_expr(expr, subs)),
        }
    }

    pub fn force_instantiate_drops(&self) {
        let drop_iface = StrId::from_static("Drop");
        let drop_method_name = StrId::from_static("drop");
        let origins: Vec<(StrId, Vec<HirType<'a, 'bump>>)> = self
            .instantiated_struct_origins
            .borrow()
            .values()
            .cloned()
            .collect();
        for (origin_name, origin_targs) in origins {
            let implements_drop = self
                .ctx
                .struct_interfaces
                .borrow()
                .get(&origin_name)
                .map(|ifaces| ifaces.contains(&drop_iface))
                .unwrap_or(false);
            if !implements_drop {
                continue;
            }
            let targs_slice = self.bump.alloc_slice(&origin_targs);
            let generic_ty = HirType::Struct {
                name: origin_name,
                field_types: &[],
                type_args: targs_slice,
            };
            self.resolve_method_for_type(&generic_ty, drop_method_name, None, &HashMap::default());
        }
    }

    pub(crate) fn try_monomorphize_struct_init_with_expected_type<'subs>(
        &self,
        value: &HirExpr<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        outer_subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<HirExpr<'a, 'bump>> {
        let HirExpr::StructInit {
            name,
            args,
            type_args: None,
            span,
        } = value
        else {
            return None;
        };
        let HirExpr::Ident(struct_name, ident_span) = &**name else {
            return None;
        };
        let HirType::Struct {
            name: expected_struct_name,
            type_args: expected_targs,
            ..
        } = expected_ty
        else {
            return None;
        };

        let names_match = *expected_struct_name == *struct_name
            || self
                .instantiated_struct_origins
                .borrow()
                .get(expected_struct_name)
                .map(|(origin, _)| *origin == *struct_name)
                .unwrap_or(false);
        if !names_match {
            return None;
        }

        if expected_targs.is_empty()
            && self
                .ctx
                .structs
                .borrow()
                .get(expected_struct_name)
                .map_or(false, |s| s.generics.is_some())
        {
            return None; // bare template type: let the old path handle it
        }

        let new_struct_name = if expected_targs.is_empty() {
            *expected_struct_name
        } else {
            let resolved_targs: Vec<HirType> = expected_targs
                .iter()
                .map(|t| {
                    let s = substitute_type(t, outer_subs, &self.bump);
                    self.instantiate_type_recursively(s, *span)
                })
                .collect();
            if resolved_targs.iter().any(contains_unresolved_generic) {
                return None;
            }
            instantiate_struct_for_types(
                self.ctx,
                &self.instantiated_structs,
                &self.instantiated_struct_origins,
                &self.instantiated_enums,
                &self.instantiated_enum_origins,
                *struct_name,
                &resolved_targs,
                &&self.bump,
            )?
            .name
        };

        // field values get their declared field type as expected type
        let field_tys: FxHashMap<StrId, HirType> = self
            .ctx
            .structs
            .borrow()
            .get(&new_struct_name)
            .map(|s| s.fields.iter().map(|f| (f.name, f.field_type)).collect())
            .unwrap_or_default();

        let new_args: Vec<HirFieldInit> = args
            .iter()
            .map(|a| HirFieldInit {
                name: a.name,
                name_span: a.name_span,
                value: match field_tys
                    .get(&a.name)
                    .filter(|t| !contains_unresolved_generic(t))
                {
                    Some(t) => self.monomorphize_expr_with_expected_type(&a.value, t, outer_subs),
                    None => self.monomorphize_expr(&a.value, outer_subs),
                },
            })
            .collect();
        let args_slice = self.bump.alloc_slice(&new_args);

        Some(HirExpr::StructInit {
            name: self
                .bump
                .alloc_value_immutable(HirExpr::Ident(new_struct_name, *ident_span)),
            args: args_slice,
            type_args: None,
            span: *span,
        })
    }

    pub(crate) fn try_monomorphize_enum_init_with_expected_type<'subs>(
        &self,
        value: &HirExpr<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        outer_subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<HirExpr<'a, 'bump>> {
        let HirExpr::EnumInit {
            enum_name,
            variant,
            args,
            type_args: None,
            span,
        } = value
        else {
            return None;
        };
        let HirType::Enum {
            name: expected_enum_name,
            type_args: expected_targs,
            variants: _,
        } = expected_ty
        else {
            return None;
        };

        let names_match = *expected_enum_name == *enum_name
            || self
                .instantiated_enum_origins
                .borrow()
                .get(expected_enum_name)
                .map(|(origin, _)| *origin == *enum_name)
                .unwrap_or(false);
        if !names_match {
            return None;
        }

        let new_enum_name = if expected_targs.is_empty() {
            *expected_enum_name
        } else {
            let resolved_targs: Vec<HirType> = expected_targs
                .iter()
                .map(|t| substitute_type(t, outer_subs, &self.bump))
                .collect();

            instantiate_enum_for_types(
                self.ctx,
                &self.instantiated_enums,
                &self.instantiated_enum_origins,
                *enum_name,
                &resolved_targs,
                &&self.bump,
            )?
            .name
        };

        let new_args: Vec<HirExpr> = args
            .iter()
            .map(|a| self.monomorphize_expr(a, outer_subs))
            .collect();
        let args_slice = self.bump.alloc_slice(&new_args);

        Some(HirExpr::EnumInit {
            enum_name: new_enum_name,
            variant: *variant,
            args: args_slice,
            type_args: None,
            span: *span,
        })
    }

    pub fn force_instantiate_allocator_frees(&self) {
        let free_method_name = StrId::from_static("free");
        let allocator_iface = StrId::from_static("Allocator");

        let mut needed: Vec<(StrId, HirType<'a, 'bump>)> = Vec::new();

        let funcs: Vec<HirFunc<'a, 'bump>> = self.functions.borrow().values().cloned().collect();

        for func in &funcs {
            let Some(body) = func.body else { continue };

            let mut param_map: FxHashMap<StrId, HirType> = FxHashMap::default();
            if let Some(params) = func.params {
                for p in params.iter() {
                    if let HirParam::Normal {
                        name, param_type, ..
                    } = p
                    {
                        param_map.insert(*name, *param_type);
                    }
                }
            }

            let this_ty = func.impl_target.map(|name| HirType::Struct {
                name,
                field_types: &[],
                type_args: &[],
            });

            let prev_params = self.current_params.replace(param_map);
            let prev_this = self.current_this.replace(this_ty);

            self.collect_owned_pointer_allocs(&body, &mut needed);

            self.current_params.replace(prev_params);
            self.current_this.replace(prev_this);
        }

        let all_structs: Vec<HirStruct<'a, 'bump>> =
            self.ctx.structs.borrow().values().cloned().collect();

        let allocator_field_of = |name: StrId| -> Option<StrId> {
            self.ctx.structs.borrow().get(&name).and_then(|s| {
                s.fields.iter().find_map(|f| match &f.field_type {
                    HirType::Struct { name: fty_name, .. } => {
                        let ifaces = self.ctx.struct_interfaces.borrow();
                        ifaces
                            .get(fty_name)
                            .map_or(false, |v| v.contains(&allocator_iface))
                            .then_some(*fty_name)
                    }
                    _ => None,
                })
            })
        };

        for hir_struct in &all_structs {
            if hir_struct.generics.is_some() {
                continue;
            }
            for f in hir_struct.fields.iter() {
                let Some((node_ty, node_name)) = owned_link_parts(&f.field_type) else {
                    continue;
                };

                let allocator_name =
                    match field_allocator_annotation(hir_struct.name, hir_struct, &f.field_type) {
                        Some(name) => Some(name),
                        None => allocator_field_of(hir_struct.name)
                            .or_else(|| allocator_field_of(node_name)),
                    };

                if let Some(allocator_name) = allocator_name {
                    needed.push((allocator_name, node_ty));
                }
            }
        }

        for (allocator_name, pointee_ty) in needed {
            if contains_unresolved_generic(&pointee_ty) {
                continue; // came from a generic template; the instantiated copy covers it
            }
            let allocator_ty = HirType::Struct {
                name: allocator_name,
                field_types: &[],
                type_args: &[],
            };

            let type_args_slice = self.bump.alloc_slice(&[pointee_ty]);
            self.resolve_method_for_type(
                &allocator_ty,
                free_method_name,
                Some(type_args_slice),
                &HashMap::default(),
            );
        }
    }

    pub(crate) fn infer_allocator_ty_from_expr(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        match expr {
            HirExpr::Block { body, .. } => match body.last() {
                Some(HirStmt::Expr(e)) => self.infer_allocator_ty_from_expr(e),
                _ => None,
            },
            HirExpr::Intrinsic {
                kind: IntrinsicKind::Own,
                args,
                ..
            } => {
                if args.len() == 3 || args.len() == 2 {
                    if let Some(ty) = self.concrete_type_of(&args[1]) {
                        return Some(ty);
                    }
                }
                self.concrete_type_of(&HirExpr::This {
                    span: Default::default(),
                })
            }

            HirExpr::Call { args, .. } | HirExpr::InterfaceCall { args, .. } => {
                let first = args.first()?;
                if let Some(ty) = self.concrete_type_of(first) {
                    return Some(ty);
                }
                self.infer_allocator_ty_from_expr(first)
            }

            HirExpr::Cast { expr, .. }
            | HirExpr::Deref { expr, .. }
            | HirExpr::Ref { expr, .. } => self.infer_allocator_ty_from_expr(expr),

            _ => None,
        }
    }

    pub(crate) fn collect_owned_pointer_allocs(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        needed: &mut Vec<(StrId, HirType<'a, 'bump>)>,
    ) {
        match stmt {
            HirStmt::Let {
                ty,
                value,
                name,
                span: _,
                ..
            } => {
                if let HirType::OwnedPointer { inner, .. } = ty {
                    if let Some(alloc_ty) = self.infer_allocator_ty_from_expr(value) {
                        if let HirType::Struct { name, .. } = peel_to_struct_owned(alloc_ty) {
                            needed.push((name, **inner));
                        }
                    }
                }
                self.collect_owned_pointer_allocs_expr(value, needed);
                self.current_params.borrow_mut().insert(*name, *ty);
            }
            HirStmt::Block { body, .. } => {
                for s in body.iter() {
                    self.collect_owned_pointer_allocs(s, needed);
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                ..
            } => {
                self.collect_owned_pointer_allocs_expr(cond, needed);
                for s in then_block.iter() {
                    self.collect_owned_pointer_allocs(s, needed);
                }
                if let Some(e) = else_block {
                    self.collect_owned_pointer_allocs(e, needed);
                }
            }
            HirStmt::While { cond, body } => {
                self.collect_owned_pointer_allocs_expr(cond, needed);
                self.collect_owned_pointer_allocs(body, needed);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.collect_owned_pointer_allocs(i, needed);
                }
                if let Some(c) = condition {
                    self.collect_owned_pointer_allocs_expr(c, needed);
                }
                if let Some(i) = increment {
                    self.collect_owned_pointer_allocs_expr(i, needed);
                }
                self.collect_owned_pointer_allocs(body, needed);
            }
            HirStmt::Return(Some(e), _) => self.collect_owned_pointer_allocs_expr(e, needed),
            HirStmt::Expr(e) => self.collect_owned_pointer_allocs_expr(e, needed),
            HirStmt::UnsafeBlock { body } => self.collect_owned_pointer_allocs(body, needed),
            HirStmt::Match { expr, arms, .. } => {
                self.collect_owned_pointer_allocs_expr(expr, needed);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_owned_pointer_allocs_expr(g, needed);
                    }
                    self.collect_owned_pointer_allocs(arm.body, needed);
                }
            }
            HirStmt::Defer(s) => self.collect_owned_pointer_allocs(s, needed),
            _ => {}
        }
    }

    pub(crate) fn collect_owned_pointer_allocs_expr(
        &self,
        expr: &HirExpr<'a, 'bump>,
        needed: &mut Vec<(StrId, HirType<'a, 'bump>)>,
    ) {
        match expr {
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.collect_owned_pointer_allocs(s, needed);
                }
            }
            HirExpr::If { if_stmt, .. } => self.collect_owned_pointer_allocs(if_stmt, needed),
            HirExpr::Match { expr, arms, .. } => {
                self.collect_owned_pointer_allocs_expr(expr, needed);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_owned_pointer_allocs_expr(g, needed);
                    }
                    self.collect_owned_pointer_allocs(arm.body, needed);
                }
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                self.collect_owned_pointer_allocs_expr(callee, needed);
                for a in args.iter() {
                    self.collect_owned_pointer_allocs_expr(a, needed);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.collect_owned_pointer_allocs_expr(left, needed);
                self.collect_owned_pointer_allocs_expr(right, needed);
            }
            HirExpr::Assignment { target, value, .. } => {
                self.collect_owned_pointer_allocs_expr(target, needed);
                self.collect_owned_pointer_allocs_expr(value, needed);
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.collect_owned_pointer_allocs_expr(object, needed);
            }
            HirExpr::Cast { expr, .. }
            | HirExpr::Deref { expr, .. }
            | HirExpr::Ref { expr, .. } => {
                self.collect_owned_pointer_allocs_expr(expr, needed);
            }
            HirExpr::Index { object, index, .. } => {
                self.collect_owned_pointer_allocs_expr(object, needed);
                self.collect_owned_pointer_allocs_expr(index, needed);
            }
            HirExpr::ArrayLiteral { elements, .. } => {
                for e in elements.iter() {
                    self.collect_owned_pointer_allocs_expr(e, needed);
                }
            }
            HirExpr::Tuple(exprs, _) => {
                for e in exprs.iter() {
                    self.collect_owned_pointer_allocs_expr(e, needed);
                }
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.collect_owned_pointer_allocs_expr(&f.value, needed);
                }
            }
            HirExpr::EnumInit { args, .. } => {
                for a in args.iter() {
                    self.collect_owned_pointer_allocs_expr(a, needed);
                }
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.collect_owned_pointer_allocs_expr(object, needed);
                self.collect_owned_pointer_allocs_expr(start, needed);
                self.collect_owned_pointer_allocs_expr(end, needed);
            }
            HirExpr::Range { start, end, .. } => {
                self.collect_owned_pointer_allocs_expr(start, needed);
                self.collect_owned_pointer_allocs_expr(end, needed);
            }
            HirExpr::ExprList { list, .. } => {
                for e in list.iter() {
                    self.collect_owned_pointer_allocs_expr(e, needed);
                }
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.collect_owned_pointer_allocs_expr(a, needed);
                }
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.collect_owned_pointer_allocs_expr(e, needed);
                    }
                }
            }
            _ => {}
        }
    }

    pub(crate) fn resolve_own_generic_method_call<'subs>(
        &self,
        new_object: &HirExpr<'a, 'bump>,
        field: StrId,
        call_type_args: Option<&'bump [HirType<'a, 'bump>]>,
        args: &[HirExpr<'a, 'bump>],
        outer_subs: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<HirExpr<'a, 'bump>> {
        let concrete_recv_ty = self.concrete_type_of(new_object)?;
        let struct_ty = peel_to_struct(&concrete_recv_ty);
        let HirType::Struct {
            name: recv_name, ..
        } = struct_ty
        else {
            return None;
        };

        let base_method_name = *self
            .ctx
            .struct_methods
            .borrow()
            .get(&recv_name)?
            .get(&field)?;
        let base_func = self.functions.borrow().get(&base_method_name)?.clone();
        let type_params = base_func.generics?;

        let is_instance_method = base_func
            .params
            .map_or(false, |p| matches!(p.first(), Some(HirParam::This { .. })));

        let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();

        if let Some(targs) = call_type_args {
            if type_params.len() != targs.len() {
                return None;
            }
            for (p, a) in type_params.iter().zip(targs.iter()) {
                let resolved = substitute_type(a, outer_subs, &self.bump);
                assert!(
                    !contains_unresolved_generic(&resolved),
                    "resolved generic for {}",
                    base_func.name
                );
                inner_subs.insert(p.name, resolved);
            }
        } else if let Some(declared_params) = base_func.params {
            let declared_types: Vec<HirType> = declared_params
                .iter()
                .filter_map(|p| match p {
                    HirParam::Normal { param_type, .. } => Some(*param_type),
                    HirParam::This { .. } => None,
                })
                .collect();
            self.infer_missing_generics(
                type_params,
                &declared_types,
                args,
                outer_subs,
                &mut inner_subs,
            );
            if inner_subs.len() != type_params.len() {
                return None;
            }
        } else {
            return None;
        }

        let mut new_args: Vec<HirExpr> = Vec::with_capacity(args.len() + 1);
        if is_instance_method {
            new_args.push(new_object.clone());
        }
        new_args.extend(args.iter().map(|a| self.monomorphize_expr(a, outer_subs)));
        let args_slice = self.bump.alloc_slice(&new_args);

        let prev_self = self.current_this.replace(Some(concrete_recv_ty));
        let new_name = self.monomorphize_function(&base_func, &inner_subs);
        self.current_this.replace(prev_self);
        let new_name = new_name?;

        Some(HirExpr::Ident(new_name, Default::default())).map(|callee_expr| HirExpr::Call {
            callee: self.bump.alloc_value_immutable(callee_expr),
            args: args_slice,
            type_args: None,
            span: Default::default(),
        })
    }

    pub(crate) fn infer_missing_generics<'subs>(
        &self,
        type_params: &[HirGeneric<'a, 'bump>],
        declared_types: &[HirType<'a, 'bump>],
        arg_exprs: &[HirExpr<'a, 'bump>],
        outer_subs: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
        inner_subs: &mut FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        for (declared_ty, arg_expr) in declared_types.iter().zip(arg_exprs.iter()) {
            if inner_subs.len() == type_params.len() {
                break;
            }
            if let Some(concrete_ty) = self.concrete_type_of(arg_expr) {
                let resolved_ty = substitute_type(&concrete_ty, outer_subs, &self.bump);
                Self::unify_generic(declared_ty, &resolved_ty, type_params, inner_subs);
            }
        }
    }

    pub(crate) fn unify_generic(
        declared: &HirType<'a, 'bump>,
        concrete: &HirType<'a, 'bump>,
        type_params: &[HirGeneric<'a, 'bump>],
        inner_subs: &mut FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        match declared {
            HirType::Generic(name) => {
                if type_params.iter().any(|p| p.name == *name) && !inner_subs.contains_key(name) {
                    inner_subs.insert(*name, *concrete);
                }
            }
            HirType::Slice(inner) => {
                if let HirType::Slice(c) = concrete {
                    Self::unify_generic(inner, c, type_params, inner_subs);
                }
            }
            HirType::Nullable(inner) => {
                if let HirType::Nullable(c) = concrete {
                    Self::unify_generic(inner, c, type_params, inner_subs);
                }
            }
            HirType::OwnedPointer { inner, .. } => {
                if let HirType::OwnedPointer { inner: c, .. } = concrete {
                    Self::unify_generic(inner, c, type_params, inner_subs);
                }
            }
            HirType::Ref { inner, .. } => {
                if let HirType::Ref { inner: c, .. } = concrete {
                    Self::unify_generic(inner, c, type_params, inner_subs);
                }
            }
            HirType::SafePointer { inner, .. } => {
                if let HirType::SafePointer { inner: c, .. } = concrete {
                    Self::unify_generic(inner, c, type_params, inner_subs);
                }
            }
            HirType::UnsafePointer { inner, .. } => {
                if let HirType::UnsafePointer { inner: c, .. } = concrete {
                    Self::unify_generic(inner, c, type_params, inner_subs);
                }
            }
            HirType::Struct {
                name: dn,
                type_args: da,
                ..
            } => {
                if let HirType::Struct {
                    name: an,
                    type_args: aa,
                    ..
                } = concrete
                {
                    if dn == an {
                        for (d, a) in da.iter().zip(aa.iter()) {
                            Self::unify_generic(d, a, type_params, inner_subs);
                        }
                    }
                }
            }
            HirType::Enum {
                name: dn,
                type_args: da,
                ..
            } => {
                if let HirType::Enum {
                    name: an,
                    type_args: aa,
                    ..
                } = concrete
                {
                    if dn == an {
                        for (d, a) in da.iter().zip(aa.iter()) {
                            Self::unify_generic(d, a, type_params, inner_subs);
                        }
                    }
                }
            }
            _ => {}
        }
    }

    /// Resolve a `ModuleAccess` callee to the mangled `StrId` of the underlying
    /// free function, if one exists in the function table.
    /// This mirrors `HirLowerer::resolve_module_function` but without the
    /// re-export guard, so it also resolves non-re-exported functions like
    /// `threads.spawn`.
    pub(crate) fn resolve_module_access_function(
        &self,
        acc: &HirModuleAccess<'a, 'bump>,
    ) -> Option<StrId> {
        if acc.member.is_empty() {
            return None;
        }
        let dg = self.ctx.dep_graph.borrow();
        let module_idx = self
            .ctx
            .imported_modules
            .borrow()
            .get(acc.path.last()?)
            .copied()
            .or_else(|| dg.resolve_module_path(acc.path))?;

        let real_idx = dg.canonical_member_module(module_idx, acc.member);
        let mangled = dg.mangle_free_function(real_idx, acc.member, false, &self.ctx.context);
        self.functions.borrow().get(&mangled).map(|f| f.name)
    }

    /// After closure-env generics (e.g. `F -> __closure_env_0`) are bound in
    /// `inner_subs`, some generic parameters (e.g. `R` in `F: func(): R`) may
    /// only appear in the *constraint* on a closure-bound generic, not in any
    /// parameter type. This method resolves those by looking up the `__call`
    /// return type on the env struct and unifying it with the constraint's
    /// return-type position.
    pub(crate) fn infer_generics_from_closure_constraints(
        &self,
        type_params: &[HirGeneric<'a, 'bump>],
        current_subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
        inner_subs: &mut FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        for param in type_params {
            if inner_subs.contains_key(&param.name) {
                continue; // already resolved
            }
            // Check if this generic appears as the return type in the
            // constraint of some already-resolved closure-env generic.
            for other_param in type_params {
                let Some(bound_ty) = current_subs.get(&other_param.name) else {
                    continue;
                };
                let HirType::Struct { name: env_name, .. } = bound_ty else {
                    continue;
                };
                // Only consider closure env structs.
                if !self.env_structs.contains_key(env_name) {
                    continue;
                }

                // Look for a Lambda constraint on `other_param` whose
                // return_type is `Generic(param.name)`.
                for constraint in other_param.constraints {
                    if let HirType::Lambda { return_type, .. } = constraint {
                        if let HirType::Generic(ret_generic) = **return_type {
                            if ret_generic == param.name {
                                // Resolve by looking up the __call return type.
                                let call_fn_name = self.env_structs.get(env_name).copied();
                                if let Some(n) = call_fn_name {
                                    let func_opt = self.functions.borrow().get(&n).cloned();
                                    if let Some(call_fn) = func_opt {
                                        if let Some(ret) = call_fn.return_type {
                                            inner_subs.insert(param.name, ret);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn owned_link_parts<'a, 'bump>(ty: &HirType<'a, 'bump>) -> Option<(HirType<'a, 'bump>, StrId)> {
    let owned: HirType<'a, 'bump> = match ty {
        HirType::Nullable(inner) => **inner,
        other => *other,
    };
    let HirType::OwnedPointer { inner, .. } = owned else {
        return None;
    };
    let pointee: HirType<'a, 'bump> = *inner;
    let HirType::Struct { name, .. } = pointee else {
        return None;
    };
    Some((pointee, name))
}

fn owned_link_target<'a, 'bump>(ty: &HirType<'a, 'bump>) -> Option<StrId> {
    owned_link_parts(ty).map(|(_, name)| name)
}

fn field_allocator_annotation<'a, 'bump>(
    struct_name: StrId,
    hir_struct: &HirStruct<'a, 'bump>,
    ty: &HirType<'a, 'bump>,
) -> Option<StrId> {
    let owned = match ty {
        HirType::Nullable(inner) => **inner,
        other => *other,
    };
    let HirType::OwnedPointer {
        allocator: Some(ann),
        ..
    } = owned
    else {
        return None;
    };

    // The only shape we know how to resolve here: `this.<field>`, a single
    // Field segment off ThisRoot. Anything else means the annotation exists

    let (hir::ProvenanceRoot::ThisRoot, [hir::ProvenancePathSegment::Field(field_name)]) =
        (ann.root, ann.path)
    else {
        panic!(
            "field_allocator_annotation: struct `{}` has an owned-pointer field with an \
             explicit allocator annotation `{}` that isn't a simple `this.<field>` path; \
             force_instantiate_allocator_frees doesn't know how to resolve this shape, so \
             the corresponding `Allocator::free<T>` would never be scheduled for \
             instantiation.",
            struct_name, ann,
        );
    };

    let field = hir_struct
        .fields
        .iter()
        .find(|f| f.name == *field_name)
        .unwrap_or_else(|| {
            panic!(
                "field_allocator_annotation: struct `{}`'s allocator annotation points to \
                 field `{}`, which doesn't exist on the struct",
                struct_name, field_name
            )
        });

    match field.field_type {
        HirType::Struct { name, .. } => Some(name),
        other => panic!(
            "field_allocator_annotation: struct `{}` field `{}` (named by an allocator \
             annotation) isn't itself a struct type -- found `{:?}`",
            struct_name, field_name, other
        ),
    }
}

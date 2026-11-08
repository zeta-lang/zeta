use ir::{
    hir::{HirExpr, HirFunc, HirParam, HirType, HirTypeKind, StrId},
    hir_utils::hir_contains_this,
    ir_hasher::FxHashMap,
    span::SourceSpan,
};

use crate::hir_lowerer::monomorphization::{
    Monomorphizer,
    assertions::{contains_unresolved_generic, peel_to_struct},
    instantiate_struct_for_types,
    monomorphizer::replace_this,
    struct_instantiation::instantiate_enum_for_types,
    substitute_type,
};

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    pub(crate) fn substitute_type_with_self(
        &self,
        ty: &HirType<'a, 'bump>,
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        let substituted = substitute_type(ty, subs, &self.bump);
        match *self.current_this.borrow() {
            Some(self_ty) if hir_contains_this(&substituted) => {
                replace_this(&substituted, &self_ty, self.bump, ty.span)
            }
            _ => substituted,
        }
    }

    pub(crate) fn instantiate_type_recursively(
        &self,
        ty: HirType<'a, 'bump>,
        span: SourceSpan<'a>,
    ) -> HirType<'a, 'bump> {
        if hir_contains_this(&ty) {
            return ty;
        }
        let ty = self.instantiate_struct_ty_if_needed(ty, span);
        let ty = self.instantiate_enum_ty_if_needed(ty);
        match ty.kind {
            HirTypeKind::Ref {
                inner,
                ref_kind: mutability_state,
                provenance,
            } => HirType {
                kind: HirTypeKind::Ref {
                    inner: self
                        .bump
                        .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                    ref_kind: mutability_state,
                    provenance,
                },
                span: ty.span,
            },
            HirTypeKind::SafePointer {
                inner,
                mutability_state,
            } => HirType {
                kind: HirTypeKind::SafePointer {
                    inner: self
                        .bump
                        .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                    mutability_state,
                },
                span: ty.span,
            },
            HirTypeKind::UnsafePointer {
                inner,
                mutability_state,
            } => HirType {
                kind: HirTypeKind::UnsafePointer {
                    inner: self
                        .bump
                        .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                    mutability_state,
                },
                span: ty.span,
            },
            HirTypeKind::OwnedPointer { inner, allocator } => HirType {
                kind: HirTypeKind::OwnedPointer {
                    inner: self
                        .bump
                        .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                    allocator,
                },
                span: ty.span,
            },
            HirTypeKind::Slice(inner) => HirType::slice(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                span,
            ),
            HirTypeKind::Array(inner, len) => HirType::array(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                len,
                span,
            ),
            HirTypeKind::Nullable(inner) => HirType::nullable(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                span,
            ),
            HirTypeKind::Tuple(elems) => {
                let new_elems: Vec<HirType> = elems
                    .iter()
                    .map(|e| self.instantiate_type_recursively(*e, span))
                    .collect();
                HirType::tuple(self.bump.alloc_slice(&new_elems), span)
            }
            _ => ty,
        }
    }

    pub(crate) fn apply_substitutions_to_func<'subs>(
        &self,
        func: &mut HirFunc<'a, 'bump>,
        substitutions: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) {
        self.apply_substitutions_to_params(func, substitutions);

        if let Some(ret) = &mut func.return_type {
            let substituted = self.substitute_type_with_self(ret, substitutions);
            *ret = self.instantiate_type_recursively(substituted, span);
        }

        func.generics = None;
    }

    pub(crate) fn apply_substitutions_to_params<'subs>(
        &self,
        func: &mut HirFunc<'a, 'bump>,
        substitutions: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        if let Some(params) = func.params {
            if params.is_empty() {
                func.params = None;
                return;
            }
            let mut new_params: Vec<HirParam> = Vec::new();
            for param in params.iter() {
                let new_param = match param {
                    HirParam::Normal {
                        name,
                        param_type,
                        multi_place,
                        span,
                    } => {
                        let substituted = self.substitute_type_with_self(param_type, substitutions);
                        HirParam::Normal {
                            name: *name,
                            param_type: self.instantiate_type_recursively(substituted, *span),
                            multi_place: *multi_place,
                            span: *span,
                        }
                    }
                    HirParam::This {
                        kind,
                        span,
                        multi_place,
                    } => HirParam::This {
                        multi_place: *multi_place,
                        kind: *kind,
                        span: *span,
                    },
                };
                new_params.push(new_param);
            }
            func.params = Some(self.bump.alloc_slice(&new_params));
        }
    }

    pub(crate) fn instantiate_struct_ty_if_needed(
        &self,
        ty: HirType<'a, 'bump>,
        span: SourceSpan<'a>,
    ) -> HirType<'a, 'bump> {
        let HirTypeKind::Struct {
            name, type_args, ..
        } = ty.kind
        else {
            return ty;
        };
        if type_args.is_empty() {
            return ty;
        }

        let normalized_args: Vec<HirType> = type_args
            .iter()
            .map(|a| self.instantiate_type_recursively(*a, span))
            .collect();
        let normalized_args = self.bump.alloc_slice(&normalized_args);

        match instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            name,
            normalized_args,
            &self.bump,
        ) {
            Some(new_struct) => {
                let field_types: Vec<HirType> =
                    new_struct.fields.iter().map(|f| f.field_type).collect();
                HirType::struct_type(
                    new_struct.name,
                    new_struct.unmangled_name,
                    self.bump.alloc_slice(&field_types),
                    &[],
                    span,
                )
            }
            None => panic!(
                "instantiate_struct_ty_if_needed: failed to instantiate `{}` with type args {:?} at {span}; \
                 this struct type would otherwise be silently left un-renamed, which produces \
                 confusing 'unknown struct' failures much later in MIR lowering. Please open an issue if you see this error message.",
                name, normalized_args
            ),
        }
    }

    fn resolve_method_for_enum(
        &mut self,
        name: StrId,
        type_args: &[HirType<'a, 'bump>],
        method_name: StrId,
        call_type_args: Option<&[HirType<'a, 'bump>]>,
        outer_subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<StrId> {
        // An instantiated enum (`Result_i64_i32`) maps back to its template + args.
        let (origin, targs): (StrId, Vec<HirType<'a, 'bump>>) = if type_args.is_empty() {
            self.instantiated_enum_origins
                .borrow()
                .get(&name)
                .cloned()
                .unwrap_or((name, Vec::new()))
        } else {
            (
                name,
                type_args
                    .iter()
                    .map(|t| substitute_type(t, outer_subs, &self.bump))
                    .collect(),
            )
        };

        let base_method_name = *self
            .ctx
            .struct_methods
            .borrow()
            .get(&origin)?
            .get(&method_name)?;
        let base_func = self.functions.borrow().get(&base_method_name)?.clone();
        let Some(type_params) = base_func.generics else {
            return Some(base_method_name);
        };

        // impl-level generics come first (see `merge_generics`), method-level follow.
        let mut inner_subs: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();
        for (p, a) in type_params.iter().zip(targs.iter()) {
            inner_subs.insert(p.name, *a);
        }
        if let Some(cta) = call_type_args {
            for (p, a) in type_params.iter().skip(targs.len()).zip(cta.iter()) {
                inner_subs.insert(p.name, substitute_type(a, outer_subs, &self.bump));
            }
        }
        if inner_subs.len() != type_params.len() {
            return None;
        }

        let this_ty = self.instantiate_type_recursively(
            HirType {
                kind: HirTypeKind::Enum {
                    name: origin,
                    unmangled_name: self.ctx.structs.borrow().get(&origin)?.unmangled_name,
                    variants: &[],
                    type_args: self.bump.alloc_slice(&targs),
                },
                span: SourceSpan::default(),
            },
            SourceSpan::default(),
        );
        let prev = self.current_this.replace(Some(this_ty));
        let result = self.monomorphize_function(&base_func, &inner_subs);
        self.current_this.replace(prev);
        result
    }

    pub(crate) fn resolve_method_for_type(
        &mut self,
        struct_ty: &HirType<'a, 'bump>,
        method_name: StrId,
        call_type_args: Option<&[HirType<'a, 'bump>]>,
        outer_subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<StrId> {
        let struct_ty = peel_to_struct(struct_ty);
        if let HirTypeKind::Enum {
            name, type_args, ..
        } = struct_ty.kind
        {
            return self.resolve_method_for_enum(
                name,
                type_args,
                method_name,
                call_type_args,
                outer_subs,
            );
        }
        let HirTypeKind::Struct {
            name,
            unmangled_name,
            type_args,
            ..
        } = struct_ty.kind
        else {
            return None;
        };

        if type_args.is_empty() {
            let origin = self
                .instantiated_struct_origins
                .borrow()
                .get(&name)
                .cloned();
            if let Some((origin_name, origin_targs)) = origin {
                let targs_slice = self.bump.alloc_slice(&origin_targs);
                let generic_ty = HirType::struct_type(
                    origin_name,
                    unmangled_name,
                    &[],
                    targs_slice,
                    SourceSpan::default(),
                );
                return self.resolve_method_for_type(
                    &generic_ty,
                    method_name,
                    call_type_args,
                    outer_subs,
                );
            }

            let base_method_name = *self
                .ctx
                .struct_methods
                .borrow()
                .get(&name)?
                .get(&method_name)?;
            let base_func = self.functions.borrow().get(&base_method_name)?.clone();

            if let Some(type_params) = base_func.generics {
                let targs = call_type_args?;
                if targs.len() != type_params.len() {
                    return None;
                }
                let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();
                for (p, a) in type_params.iter().zip(targs.iter()) {
                    let resolved = substitute_type(a, outer_subs, &self.bump);

                    assert!(
                        !contains_unresolved_generic(&resolved),
                        "resolved generic for {}",
                        base_func.name
                    );

                    inner_subs.insert(p.name, resolved);
                }
                let prev_self = self.current_this.replace(Some(*struct_ty));
                let result = self.monomorphize_function(&base_func, &inner_subs);
                self.current_this.replace(prev_self);
                return result;
            }

            if let Some(type_params) = base_func.generics {
                let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();
                if let Some(targs) = call_type_args {
                    if targs.len() != type_params.len() {
                        return None;
                    }
                    for (p, a) in type_params.iter().zip(targs.iter()) {
                        inner_subs.insert(p.name, substitute_type(a, outer_subs, &self.bump));
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
                        &[],
                        outer_subs,
                        &mut inner_subs,
                    );
                    if inner_subs.len() != type_params.len() {
                        return None;
                    }
                } else {
                    return None;
                }
                let prev_self = self.current_this.replace(Some(struct_ty.clone()));
                let result = self.monomorphize_function(&base_func, &inner_subs);
                self.current_this.replace(prev_self);
                return result;
            }

            return Some(base_method_name);
        }

        let instantiated = instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            name,
            type_args,
            &self.bump,
        )?;
        let concrete_name = instantiated.name;

        let base_method_name = *self
            .ctx
            .struct_methods
            .borrow()
            .get(&name)?
            .get(&method_name)?;
        let base_func = self.functions.borrow().get(&base_method_name)?.clone();
        let struct_generics = self
            .ctx
            .structs
            .borrow()
            .get(&name)
            .and_then(|s| s.generics.clone());
        let type_params = base_func.generics.as_ref().or(struct_generics.as_ref());
        let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();
        if let Some(type_params) = type_params {
            if type_params.len() != type_args.len() {
                return None;
            }
            for (p, a) in type_params.iter().zip(type_args.iter()) {
                inner_subs.insert(p.name, a.clone());
            }
        }

        let field_types: Vec<HirType> = instantiated.fields.iter().map(|f| f.field_type).collect();
        let concrete_recv_ty = HirType::struct_type(
            concrete_name,
            instantiated.unmangled_name,
            self.bump.alloc_slice(&field_types),
            &[],
            SourceSpan::default(),
        );

        let prev_self = self.current_this.replace(Some(concrete_recv_ty));
        let result = self.monomorphize_function(&base_func, &inner_subs);
        self.current_this.replace(prev_self);
        result
    }

    pub(crate) fn resolve_struct_key(&self, name: StrId) -> StrId {
        if self.ctx.structs.borrow().contains_key(&name) {
            return name;
        }
        self.ctx
            .resolve_type_path_name(&[], name, SourceSpan::default())
    }

    pub(crate) fn concrete_type_of(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        match expr {
            HirExpr::StructInit {
                name, type_args, ..
            } => {
                let HirExpr::Ident(n, _) = &**name else {
                    return None;
                };
                // TODO: this should give field types, and unmangled names. This is a bug that affects `sizeof` and `type_name`
                let s_key = self.resolve_struct_key(*n);
                Some(HirType::struct_type(
                    s_key,
                    s_key,
                    &[],
                    type_args.unwrap_or(&[]),
                    expr.get_span(),
                ))
            }
            HirExpr::Call { callee, .. } => match &**callee {
                HirExpr::Ident(name, _) => self
                    .functions
                    .borrow()
                    .get(name)
                    .and_then(|f| f.return_type)
                    .filter(|t| !contains_unresolved_generic(t) && !hir_contains_this(t)),
                _ => None,
            },
            HirExpr::This { .. } => *self.current_this.borrow(),
            HirExpr::Ident(name, _) => self
                .current_params
                .borrow()
                .get(name)
                .copied()
                .or_else(|| self.ctx.variable_types.borrow().get(name).copied())
                .or_else(|| {
                    let resolved_name = self.resolve_struct_key(*name);

                    if let Some(self_ty) = *self.current_this.borrow() {
                        let struct_ty = peel_to_struct(&self_ty);
                        if let HirTypeKind::Struct {
                            name: cur_struct_name,
                            ..
                        } = struct_ty.kind
                        {
                            if cur_struct_name == resolved_name {
                                return Some(self_ty);
                            }
                            let origin = self
                                .instantiated_struct_origins
                                .borrow()
                                .get(&cur_struct_name)
                                .cloned();
                            if let Some((origin, _)) = origin {
                                if origin == resolved_name {
                                    return Some(self_ty);
                                }
                            }
                        }
                    }
                    if let Some(s_ty) = self.ctx.structs.borrow().get(&resolved_name) {
                        return Some(HirType::struct_type(
                            resolved_name,
                            s_ty.unmangled_name,
                            self.ctx.bump.alloc_slice(
                                s_ty.fields
                                    .iter()
                                    .map(|f| f.field_type)
                                    .collect::<Vec<_>>()
                                    .as_slice(),
                            ),
                            &[],
                            expr.get_span(),
                        ));
                    }
                    None
                }),
            HirExpr::Deref { expr, .. } => {
                let inner = self.concrete_type_of(expr)?;
                match inner.kind {
                    HirTypeKind::Ref { inner, .. }
                    | HirTypeKind::SafePointer { inner, .. }
                    | HirTypeKind::UnsafePointer { inner, .. }
                    | HirTypeKind::OwnedPointer { inner, .. } => Some(*inner),
                    _ => Some(inner),
                }
            }
            HirExpr::Ref { expr, .. } => self.concrete_type_of(expr),
            HirExpr::Cast { target_type, .. } => Some(*target_type),
            HirExpr::ModuleAccess(acc) => {
                let (&struct_name, module_path) = acc.path.split_last()?;
                let target_key = if self.ctx.structs.borrow().contains_key(&struct_name) {
                    struct_name
                } else {
                    self.ctx
                        .resolve_type_path_name(module_path, struct_name, acc.span)
                };
                if let Some(self_ty) = *self.current_this.borrow() {
                    let struct_ty = peel_to_struct(&self_ty);
                    if let HirTypeKind::Struct {
                        name: cur_struct_name,
                        ..
                    } = struct_ty.kind
                    {
                        if cur_struct_name == target_key {
                            return Some(self_ty);
                        }
                        if let Some((origin, _)) = self
                            .instantiated_struct_origins
                            .borrow()
                            .get(&cur_struct_name)
                            .cloned()
                        {
                            if origin == target_key {
                                return Some(self_ty);
                            }
                        }
                    }
                }
                if let Some(s_ty) = self.ctx.structs.borrow().get(&target_key) {
                    return Some(HirType::struct_type(
                        target_key,
                        s_ty.unmangled_name,
                        self.ctx.bump.alloc_slice(
                            s_ty.fields
                                .iter()
                                .map(|f| f.field_type)
                                .collect::<Vec<_>>()
                                .as_slice(),
                        ),
                        &[],
                        expr.get_span(),
                    ));
                }
                None
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let obj_ty = self.concrete_type_of(object)?;
                self.field_type_of(&obj_ty, *field)
            }
            _ => None,
        }
    }

    pub(crate) fn field_type_of(
        &self,
        ty: &HirType<'a, 'bump>,
        field: StrId,
    ) -> Option<HirType<'a, 'bump>> {
        let struct_name = match ty.kind {
            HirTypeKind::Struct { name, .. } => name,
            HirTypeKind::Ref { inner, .. }
            | HirTypeKind::SafePointer { inner, .. }
            | HirTypeKind::OwnedPointer { inner, .. }
            | HirTypeKind::UnsafePointer { inner, .. } => return self.field_type_of(inner, field),
            _ => return None,
        };

        let structs = self.ctx.structs.borrow();
        let hir_struct = structs.get(&struct_name)?;
        let field_def = hir_struct.fields.iter().find(|f| f.name == field)?;
        Some(field_def.field_type)
    }

    pub(crate) fn instantiate_enum_ty_if_needed(
        &self,
        ty: HirType<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let HirTypeKind::Enum {
            name, type_args, ..
        } = ty.kind
        else {
            return ty;
        };
        if type_args.is_empty() {
            return ty;
        }

        let normalized: Vec<HirType> = type_args
            .iter()
            .map(|a| self.instantiate_type_recursively(*a, Default::default()))
            .collect();
        let normalized = self.bump.alloc_slice(&normalized);

        let Some(new_enum) = instantiate_enum_for_types(
            self.ctx,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            name,
            normalized,
            &self.bump,
        ) else {
            return ty;
        };
        // TODO: this should give type_args. This is a bug that affects `type_name`
        HirType {
            kind: HirTypeKind::Enum {
                name: new_enum.name,
                unmangled_name: new_enum.unmangled_name,
                type_args: &[],
                variants: new_enum.variants,
            },
            span: ty.span,
        }
    }

    pub fn closure_env_type(&self, env_name: StrId) -> HirType<'a, 'bump> {
        let field_types: Vec<HirType<'a, 'bump>> = self
            .ctx
            .structs
            .borrow()
            .get(&env_name)
            .map(|s| s.fields.iter().map(|f| f.field_type).collect())
            .unwrap_or_default();
        HirType::struct_type(
            env_name,
            env_name, // closure
            self.bump.alloc_slice(&field_types),
            &[], // closure
            SourceSpan::default(),
        )
    }
}

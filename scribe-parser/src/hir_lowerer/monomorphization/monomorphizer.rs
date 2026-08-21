use ir::span::SourceSpan;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use crate::hir_lowerer::LoweringCtx;
use crate::hir_lowerer::monomorphization::instantiate_struct_for_types;
use crate::hir_lowerer::monomorphization::struct_instantiation::instantiate_enum_for_types;

use super::naming::suffix_for_subs;
use super::type_substitution::substitute_type;
use ir::hir::{
    Hir, HirExpr, HirFieldInit, HirFunc, HirGeneric, HirMatchArm, HirModule, HirParam, HirStmt,
    HirType, InterpolationPart, StrId,
};
use ir::ir_hasher::{FxHashMap, HashMap};
use zetaruntime::arena::GrowableAtomicBump;
use zetaruntime::string_pool::StringPool;

fn contains_unresolved_generic(ty: &HirType) -> bool {
    match ty {
        HirType::Generic(_) => true,
        HirType::Slice(inner) | HirType::Array(inner, _) => contains_unresolved_generic(inner),
        HirType::Nullable(inner) => contains_unresolved_generic(inner),
        HirType::Ref { inner, .. }
        | HirType::SafePointer { inner, .. }
        | HirType::UnsafePointer { inner, .. }
        | HirType::OwnedPointer { inner, .. } => contains_unresolved_generic(inner),
        HirType::Struct {
            field_types,
            type_args,
            ..
        } => {
            field_types.iter().any(contains_unresolved_generic)
                || type_args.iter().any(contains_unresolved_generic)
        }
        HirType::DynInterface(_, args)
        | HirType::Enum {
            type_args: args, ..
        } => args.iter().any(contains_unresolved_generic),
        HirType::Lambda {
            params,
            return_type,
        } => {
            params.iter().any(contains_unresolved_generic)
                || contains_unresolved_generic(return_type)
        }
        HirType::Tuple(elems) => elems.iter().any(contains_unresolved_generic),
        HirType::Dyn { bounds } => bounds.iter().any(contains_unresolved_generic),
        _ => false,
    }
}

pub(super) fn peel_to_struct_owned<'a, 'bump>(ty: HirType<'a, 'bump>) -> HirType<'a, 'bump> {
    match ty {
        HirType::Ref { inner, .. }
        | HirType::SafePointer { inner, .. }
        | HirType::OwnedPointer { inner, .. }
        | HirType::UnsafePointer { inner, .. } => peel_to_struct_owned(*inner),
        _ => ty,
    }
}

pub(super) fn peel_to_struct<'b, 'a, 'bump>(ty: &'b HirType<'a, 'bump>) -> &'b HirType<'a, 'bump> {
    match ty {
        HirType::Ref { inner, .. }
        | HirType::SafePointer { inner, .. }
        | HirType::OwnedPointer { inner, .. }
        | HirType::UnsafePointer { inner, .. } => peel_to_struct(inner),
        _ => ty,
    }
}

pub struct Monomorphizer<'a, 'bump, 'ctx> {
    pub instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub instantiated_structs: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub instantiated_struct_origins:
        Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    pub instantiated_enums: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    pub instantiated_enum_origins: Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    current_return_type: RefCell<Option<HirType<'a, 'bump>>>,
    functions: Rc<RefCell<FxHashMap<StrId, HirFunc<'a, 'bump>>>>,
    bump: Arc<GrowableAtomicBump<'bump>>,
    context: Arc<StringPool>,
    ctx: &'ctx mut LoweringCtx<'a, 'bump>,
    current_this: RefCell<Option<HirType<'a, 'bump>>>,
    _phantom: PhantomData<&'bump ()>,
    current_params: RefCell<FxHashMap<StrId, HirType<'a, 'bump>>>,
}

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    pub fn new(
        context: Arc<StringPool>,
        bump: Arc<GrowableAtomicBump<'bump>>,
        functions: Rc<RefCell<FxHashMap<StrId, HirFunc<'a, 'bump>>>>,
        ctx: &'ctx mut LoweringCtx<'a, 'bump>,
        instantiated_functions: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_structs: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_struct_origins: Rc<
            RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>,
        >,
        instantiated_enums: Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
        instantiated_enum_origins: Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    ) -> Self {
        Self {
            instantiated_functions,
            instantiated_structs,
            instantiated_enums,
            instantiated_enum_origins,
            instantiated_struct_origins,
            functions,
            bump,
            context,
            ctx,
            current_this: RefCell::new(None),
            current_return_type: RefCell::new(None),
            _phantom: PhantomData,
            current_params: RefCell::new(FxHashMap::default()),
        }
    }

    pub fn run(&mut self, module: HirModule<'a, 'bump>) -> HirModule<'a, 'bump> {
        self.ctx.named_imports.borrow_mut().clear();
        self.ctx.imported_modules.borrow_mut().clear();
        for import_path in module.imports {
            let Some(target_idx) = self
                .ctx
                .dep_graph
                .borrow()
                .resolve_module_path(import_path.path)
            else {
                continue;
            };
            match import_path.member {
                None => {
                    if let Some(&last) = import_path.path.last() {
                        self.ctx
                            .imported_modules
                            .borrow_mut()
                            .insert(last, target_idx);
                    }
                }
                Some(member) => {
                    self.ctx
                        .named_imports
                        .borrow_mut()
                        .insert(member, target_idx);
                }
            }
        }

        let empty_subs: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();
        let mut new_items: Vec<Hir<'a, 'bump>> = Vec::with_capacity(module.items.len());

        for item in module.items {
            match item {
                Hir::Func(f) => {
                    let mut new_func = (*f).clone();
                    if new_func.generics.is_none() {
                        if let Some(body) = new_func.body {
                            let prev_module_idx = self.ctx.module_idx;
                            self.ctx.module_idx = f.declaring_module_idx;

                            let prev_return_type =
                                self.current_return_type.replace(new_func.return_type);
                            let new_body = self.monomorphize_stmt(&body, &empty_subs);
                            self.current_return_type.replace(prev_return_type);
                            new_func.body = Some(*self.bump.alloc_value_immutable(new_body));

                            self.ctx.module_idx = prev_module_idx;
                        }
                    }
                    new_items.push(Hir::Func(self.bump.alloc_value_immutable(new_func)));
                }
                Hir::Interface(iface) => {
                    let mut new_iface = (*iface).clone();
                    if new_iface.generics.is_none() {
                        if let Some(methods) = new_iface.methods {
                            let mut new_methods = Vec::with_capacity(methods.len());
                            for m in methods.iter() {
                                let mut nm = m.clone();
                                if nm.generics.is_none() {
                                    if let Some(params) = nm.params {
                                        let new_params: Vec<HirParam> = params
                                            .iter()
                                            .map(|p| match p {
                                                HirParam::Normal {
                                                    name,
                                                    param_type,
                                                    span,
                                                } => HirParam::Normal {
                                                    name: *name,
                                                    param_type: self
                                                        .instantiate_type_recursively(*param_type),
                                                    span: *span,
                                                },
                                                HirParam::This { kind, span } => HirParam::This {
                                                    kind: *kind,
                                                    span: *span,
                                                },
                                            })
                                            .collect();
                                        nm.params =
                                            Some(self.bump.alloc_slice_immutable(&new_params));
                                    }
                                    nm.return_type = nm
                                        .return_type
                                        .map(|rt| self.instantiate_type_recursively(rt));
                                }
                                new_methods.push(nm);
                            }
                            new_iface.methods = Some(self.bump.alloc_slice(&new_methods));
                        }
                    }
                    new_items.push(Hir::Interface(self.bump.alloc_value_immutable(new_iface)));
                }
                Hir::Impl(i) => {
                    let mut new_impl = (*i).clone();
                    if new_impl.generics.is_none() {
                        if let Some(methods) = new_impl.methods {
                            let mut new_methods = Vec::new();
                            for m in methods.iter() {
                                let mut nm = m.clone();

                                if let Some(params) = nm.params {
                                    let new_params: Vec<HirParam> = params
                                        .iter()
                                        .map(|p| match p {
                                            HirParam::Normal {
                                                name,
                                                param_type,
                                                span,
                                            } => {
                                                let substituted = substitute_type(
                                                    param_type,
                                                    &empty_subs,
                                                    self.bump.clone(),
                                                );
                                                HirParam::Normal {
                                                    name: *name,
                                                    param_type: self
                                                        .instantiate_type_recursively(substituted),
                                                    span: *span,
                                                }
                                            }
                                            HirParam::This { kind, span } => HirParam::This {
                                                kind: *kind,
                                                span: *span,
                                            },
                                        })
                                        .collect();
                                    nm.params = Some(self.bump.alloc_slice_immutable(&new_params));
                                }

                                if let Some(body) = nm.body {
                                    let prev_module_idx = self.ctx.module_idx;
                                    self.ctx.module_idx = m.declaring_module_idx;

                                    let return_type_for_body = nm.return_type.map(|ret_ty| {
                                        substitute_type(&ret_ty, &empty_subs, self.bump.clone())
                                    });

                                    let prev_return_type =
                                        self.current_return_type.replace(return_type_for_body);
                                    let new_body = self.monomorphize_stmt(&body, &empty_subs);
                                    self.current_return_type.replace(prev_return_type);

                                    nm.body = Some(*self.bump.alloc_value_immutable(new_body));
                                    nm.return_type = return_type_for_body
                                        .map(|ty| self.instantiate_type_recursively(ty));

                                    self.ctx.module_idx = prev_module_idx;
                                }
                                new_methods.push(nm);
                            }
                            new_impl.methods = Some(self.bump.alloc_slice(&new_methods));
                        }
                    }
                    new_items.push(Hir::Impl(self.bump.alloc_value_immutable(new_impl)));
                }
                other => new_items.push(*other),
            }
        }

        new_items.retain(|hir| match hir {
            Hir::Func(hir_func) => hir_func.generics.is_none(),
            Hir::Struct(hir_struct) => hir_struct.generics.is_none(),
            Hir::Interface(hir_interface) => hir_interface.generics.is_none(),
            Hir::Impl(hir_impl) => hir_impl.generics.is_none(),
            Hir::Enum(hir_enum) => hir_enum.generics.is_none(),
            _ => true,
        });

        self.assert_fully_monomorphized(&new_items);

        HirModule {
            name: module.name,
            imports: module.imports,
            items: self.bump.alloc_slice(&new_items),
        }
    }

    fn instantiate_enum_ty_if_needed(&self, ty: HirType<'a, 'bump>) -> HirType<'a, 'bump> {
        let HirType::Enum {
            name,
            type_args,
            variants: _,
        } = &ty
        else {
            return ty;
        };
        if type_args.is_empty() {
            return ty;
        }
        let Some(new_enum) = instantiate_enum_for_types(
            self.ctx,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            *name,
            type_args,
            self.bump.clone(),
        ) else {
            return ty;
        };
        HirType::Enum {
            name: new_enum.name,
            type_args: &[],
            variants: new_enum.variants,
        }
    }

    fn resolve_own_generic_method_call<'subs>(
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
                let resolved = substitute_type(a, outer_subs, self.bump.clone());
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

    fn infer_missing_generics<'subs>(
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
                let resolved_ty = substitute_type(&concrete_ty, outer_subs, self.bump.clone());
                Self::unify_generic(declared_ty, &resolved_ty, type_params, inner_subs);
            }
        }
    }

    fn unify_generic(
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
            _ => {}
        }
    }

    fn resolve_struct_key(&self, name: StrId) -> StrId {
        if self.ctx.structs.borrow().contains_key(&name) {
            return name;
        }
        self.ctx
            .resolve_type_path_name(&[], name, SourceSpan::default())
    }

    fn concrete_type_of(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        match expr {
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
                        if let HirType::Struct {
                            name: cur_struct_name,
                            ..
                        } = struct_ty
                        {
                            if *cur_struct_name == resolved_name {
                                return Some(self_ty);
                            }
                            if let Some((origin, _)) = self
                                .instantiated_struct_origins
                                .borrow()
                                .get(&cur_struct_name)
                                .cloned()
                            {
                                if origin == resolved_name {
                                    return Some(self_ty);
                                }
                            }
                        }
                    }
                    if self.ctx.structs.borrow().contains_key(&resolved_name) {
                        return Some(HirType::Struct {
                            name: resolved_name,
                            field_types: &[],
                            type_args: &[],
                        });
                    }
                    None
                }),
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
                    if let HirType::Struct {
                        name: cur_struct_name,
                        ..
                    } = struct_ty
                    {
                        if *cur_struct_name == target_key {
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
                if self.ctx.structs.borrow().contains_key(&target_key) {
                    return Some(HirType::Struct {
                        name: target_key,
                        field_types: &[],
                        type_args: &[],
                    });
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

    fn field_type_of(&self, ty: &HirType<'a, 'bump>, field: StrId) -> Option<HirType<'a, 'bump>> {
        let struct_name = match ty {
            HirType::Struct { name, .. } => *name,
            HirType::Ref { inner, .. }
            | HirType::SafePointer { inner, .. }
            | HirType::OwnedPointer { inner, .. }
            | HirType::UnsafePointer { inner, .. } => return self.field_type_of(inner, field),
            _ => return None,
        };

        let structs = self.ctx.structs.borrow();
        let hir_struct = structs.get(&struct_name)?;
        let field_def = hir_struct.fields.iter().find(|f| f.name == field)?;
        Some(field_def.field_type)
    }

    fn resolve_method_for_type(
        &self,
        struct_ty: &HirType<'a, 'bump>,
        method_name: StrId,
        call_type_args: Option<&[HirType<'a, 'bump>]>,
        outer_subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<StrId> {
        let struct_ty = peel_to_struct(struct_ty);
        let HirType::Struct {
            name, type_args, ..
        } = struct_ty
        else {
            return None;
        };

        if type_args.is_empty() {
            if let Some((origin_name, origin_targs)) =
                self.instantiated_struct_origins.borrow().get(name).cloned()
            {
                let targs_slice = self.bump.alloc_slice_immutable(&origin_targs);
                let generic_ty = HirType::Struct {
                    name: origin_name,
                    field_types: &[],
                    type_args: targs_slice,
                };
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
                .get(name)?
                .get(&method_name)?;
            let base_func = self.functions.borrow().get(&base_method_name)?.clone();

            if let Some(type_params) = base_func.generics {
                let targs = call_type_args?;
                if targs.len() != type_params.len() {
                    return None;
                }
                let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();
                for (p, a) in type_params.iter().zip(targs.iter()) {
                    let resolved = substitute_type(a, outer_subs, self.bump.clone());

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

            return Some(base_method_name);
        }

        let instantiated = instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            *name,
            type_args,
            self.bump.clone(),
        )?;
        let concrete_name = instantiated.name;

        let base_method_name = *self
            .ctx
            .struct_methods
            .borrow()
            .get(name)?
            .get(&method_name)?;
        let base_func = self.functions.borrow().get(&base_method_name)?.clone();
        let struct_generics = self
            .ctx
            .structs
            .borrow()
            .get(name)
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
        let concrete_recv_ty = HirType::Struct {
            name: concrete_name,
            field_types: self.bump.alloc_slice_immutable(&field_types),
            type_args: &[],
        };

        let prev_self = self.current_this.replace(Some(concrete_recv_ty));
        let result = self.monomorphize_function(&base_func, &inner_subs);
        self.current_this.replace(prev_self);
        result
    }

    pub fn monomorphize_function<'subs>(
        &self,
        func: &HirFunc<'a, 'bump>,
        substitutions: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<StrId> {
        if let Some(type_params) = func.generics {
            if let Some(missing) = type_params
                .iter()
                .find(|p| !substitutions.contains_key(&p.name))
            {
                panic!(
                    "cannot monomorphize `{}`: type parameter `{}` has no substitution (have: {:?})",
                    func.name,
                    missing.name,
                    substitutions.keys().collect::<Vec<_>>()
                );
            }
        }

        if let Some((bad_param, bad_ty)) = substitutions
            .iter()
            .find(|(_, ty)| contains_unresolved_generic(ty))
        {
            panic!(
                "cannot monomorphize `{}`: substitution for type parameter `{}` is still generic ({:?}). \
                 This means an unresolved generic leaked into `substitutions` from an outer scope \
                 without being resolved via `substitute_type(_, outer_subs, ...)` first.",
                func.name, bad_param, bad_ty
            );
        }

        let mut new_func = func.clone();
        self.apply_substitutions_to_func(&mut new_func, substitutions);

        if new_func.impl_target.is_some() {
            let cur_self = *self.current_this.borrow();
            if let Some(ref self_ty) = cur_self {
                if let HirType::Struct { name, .. } = peel_to_struct(self_ty) {
                    new_func.impl_target = Some(*name);
                }
            }
        }

        let suffix = suffix_for_subs(self.context.clone(), substitutions);
        let orig_name = new_func.name.clone();

        let key = (orig_name.clone(), suffix.clone());
        {
            let instantiated_functions = self.instantiated_functions.borrow();
            if let Some(existing) = instantiated_functions.get(&key) {
                return Some(*existing);
            }
        }

        const UNDERSCORE_LEN: usize = 1;
        let mut small_vec: SmallVec<u8, 32> =
            SmallVec::with_capacity(orig_name.len() + UNDERSCORE_LEN + suffix.len());

        small_vec.extend_from_slice(self.context.resolve_bytes(&*orig_name));
        small_vec.push(b'_');
        small_vec.extend_from_slice(self.context.resolve_bytes(&*suffix));

        let new_name = StrId(self.context.intern_bytes(small_vec.as_slice()));

        new_func.name = new_name;

        let mut param_map: FxHashMap<StrId, HirType> = FxHashMap::default();
        if let Some(params) = new_func.params {
            for p in params.iter() {
                if let HirParam::Normal {
                    name, param_type, ..
                } = p
                {
                    param_map.insert(*name, *param_type);
                }
            }
        }
        let prev_params = self.current_params.replace(param_map);
        let prev_return_type = self.current_return_type.replace(new_func.return_type);

        if let Some(body) = new_func.body {
            let new_body = self.monomorphize_stmt(&body, substitutions);
            new_func.body = Some(*self.bump.alloc_value_immutable(new_body));
        }

        self.current_params.replace(prev_params);
        self.current_return_type.replace(prev_return_type);

        self.functions
            .borrow_mut()
            .insert(new_func.name.clone(), new_func);

        self.instantiated_functions
            .borrow_mut()
            .insert(key, new_name);

        Some(new_name)
    }

    fn substitute_type_with_self(
        &self,
        ty: &HirType<'a, 'bump>,
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        if let HirType::This = ty {
            if let Some(self_ty) = *self.current_this.borrow() {
                return self_ty;
            }
        }
        substitute_type(ty, subs, self.bump.clone())
    }

    fn instantiate_type_recursively(&self, ty: HirType<'a, 'bump>) -> HirType<'a, 'bump> {
        let ty = self.instantiate_struct_ty_if_needed(ty);
        let ty = self.instantiate_enum_ty_if_needed(ty);
        match ty {
            HirType::Ref {
                inner,
                mutability_state,
                provenance,
            } => HirType::Ref {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
                mutability_state,
                provenance,
            },
            HirType::SafePointer {
                inner,
                mutability_state,
            } => HirType::SafePointer {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
                mutability_state,
            },
            HirType::UnsafePointer {
                inner,
                mutability_state,
            } => HirType::UnsafePointer {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
                mutability_state,
            },
            HirType::OwnedPointer { inner, allocator } => HirType::OwnedPointer {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
                allocator,
            },
            HirType::Slice(inner) => HirType::Slice(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
            ),
            HirType::Array(inner, len) => HirType::Array(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
                len,
            ),
            HirType::Nullable(inner) => HirType::Nullable(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner)),
            ),
            HirType::Tuple(elems) => {
                let new_elems: Vec<HirType> = elems
                    .iter()
                    .map(|e| self.instantiate_type_recursively(*e))
                    .collect();
                HirType::Tuple(self.bump.alloc_slice_immutable(&new_elems))
            }
            other => other,
        }
    }

    fn apply_substitutions_to_func<'subs>(
        &self,
        func: &mut HirFunc<'a, 'bump>,
        substitutions: &'subs FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        self.apply_substitutions_to_params(func, substitutions);

        if let Some(ret) = &mut func.return_type {
            let substituted = self.substitute_type_with_self(ret, substitutions);
            *ret = self.instantiate_type_recursively(substituted);
        }

        func.generics = None;
    }

    fn apply_substitutions_to_params<'subs>(
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
                        span,
                    } => {
                        let substituted = self.substitute_type_with_self(param_type, substitutions);
                        HirParam::Normal {
                            name: *name,
                            param_type: self.instantiate_type_recursively(substituted),
                            span: *span,
                        }
                    }
                    HirParam::This { kind, span } => HirParam::This {
                        kind: *kind,
                        span: *span,
                    },
                };
                new_params.push(new_param);
            }
            func.params = Some(self.bump.alloc_slice_immutable(&new_params));
        }
    }

    pub fn monomorphize_stmt<'subs>(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        substitutions: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirStmt<'a, 'bump> {
        match stmt {
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                let new_init = init.map(|i| {
                    let s = self.monomorphize_stmt(i, substitutions);
                    self.bump.alloc_value_immutable(s)
                });
                let new_cond = condition.map(|c| {
                    let e = self.monomorphize_expr(c, substitutions);
                    self.bump.alloc_value_immutable(e)
                });
                let new_incr = increment.map(|i| {
                    let e = self.monomorphize_expr(i, substitutions);
                    self.bump.alloc_value_immutable(e)
                });
                let new_body = self.monomorphize_stmt(body, substitutions);
                HirStmt::For {
                    init: new_init,
                    condition: new_cond,
                    increment: new_incr,
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
            } => {
                let new_cond = self.monomorphize_expr(cond, substitutions);
                let new_then: Vec<HirStmt> = then_block
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, substitutions))
                    .collect();
                let then_slice = self.bump.alloc_slice(&new_then);
                let new_else = else_block.map(|e| {
                    let new_stmt = self.monomorphize_stmt(e, substitutions);
                    self.bump.alloc_value_immutable(new_stmt)
                });
                HirStmt::If {
                    cond: *self.bump.alloc_value_immutable(new_cond),
                    then_block: then_slice,
                    else_block: new_else,
                }
            }
            HirStmt::While { cond, body } => {
                let new_cond = self.monomorphize_expr(cond, substitutions);
                let new_body = self.monomorphize_stmt(body, substitutions);
                HirStmt::While {
                    cond: self.bump.alloc_value_immutable(new_cond),
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::Let {
                name,
                ty,
                value,
                mutable,
                is_static,
                catch_pattern,
                else_block,
                span,
            } => {
                let subd_ty = substitute_type(ty, substitutions, self.bump.clone());

                let new_value = self
                    .try_monomorphize_assoc_call(value, &subd_ty, substitutions)
                    .unwrap_or_else(|| {
                        self.monomorphize_expr_with_expected_type(value, &subd_ty, substitutions)
                    });

                let new_ty = self.instantiate_struct_ty_if_needed(subd_ty);
                let new_ty = self.instantiate_enum_ty_if_needed(new_ty);
                self.ctx.variable_types.borrow_mut().insert(*name, new_ty);

                HirStmt::Let {
                    name: *name,
                    ty: new_ty,
                    value: *self.bump.alloc_value_immutable(new_value),
                    mutable: *mutable,
                    is_static: *is_static,
                    catch_pattern: *catch_pattern,
                    else_block: *else_block,
                    span: *span,
                }
            }
            HirStmt::Return(opt) => {
                let new_opt = opt.map(|e| {
                    let ret_ty = *self.current_return_type.borrow();
                    let new_expr = match ret_ty {
                        Some(rt) => {
                            self.monomorphize_expr_with_expected_type(e, &rt, substitutions)
                        }
                        None => self.monomorphize_expr(e, substitutions),
                    };
                    self.bump.alloc_value_immutable(new_expr)
                });
                HirStmt::Return(new_opt)
            }
            HirStmt::Expr(e) => {
                let new_expr = self.monomorphize_expr(e, substitutions);
                HirStmt::Expr(self.bump.alloc_value_immutable(new_expr))
            }
            HirStmt::UnsafeBlock { body } => {
                let new_body = self.monomorphize_stmt(body, substitutions);
                HirStmt::UnsafeBlock {
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            HirStmt::Block { body } => {
                let new_body: Vec<HirStmt> = body
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, substitutions))
                    .collect();
                HirStmt::Block {
                    body: self.bump.alloc_slice(&new_body),
                }
            }
            HirStmt::Match { expr, arms } => {
                let new_expr = self.monomorphize_expr(expr, substitutions);
                let new_arms: Vec<HirMatchArm> = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.map(|g| {
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(g, substitutions))
                        }),
                        body: self
                            .bump
                            .alloc_value_immutable(self.monomorphize_stmt(arm.body, substitutions)),
                    })
                    .collect();
                HirStmt::Match {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                }
            }
            HirStmt::Defer(stmt) => {
                let new_stmt = self.monomorphize_stmt(stmt, substitutions);
                HirStmt::Defer(self.bump.alloc_value_immutable(new_stmt))
            }

            _ => stmt.clone(),
        }
    }

    pub fn force_instantiate_drops(&self) {
        let drop_iface = StrId(self.context.intern("Drop"));
        let drop_method_name = StrId(self.context.intern("drop"));
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
            let targs_slice = self.bump.alloc_slice_immutable(&origin_targs);
            let generic_ty = HirType::Struct {
                name: origin_name,
                field_types: &[],
                type_args: targs_slice,
            };
            self.resolve_method_for_type(&generic_ty, drop_method_name, None, &HashMap::default());
        }
    }

    fn try_monomorphize_enum_init_with_expected_type<'subs>(
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
                .map(|t| substitute_type(t, outer_subs, self.bump.clone()))
                .collect();

            instantiate_enum_for_types(
                self.ctx,
                &self.instantiated_enums,
                &self.instantiated_enum_origins,
                *enum_name,
                &resolved_targs,
                self.bump.clone(),
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

    fn try_monomorphize_assoc_call<'subs>(
        &self,
        value: &HirExpr<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        outer_subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> Option<HirExpr<'a, 'bump>> {
        let HirExpr::Call {
            callee,
            args,
            type_args: None,
            span,
        } = value
        else {
            return None;
        };
        let HirExpr::ModuleAccess(acc) = &**callee else {
            return None;
        };
        let HirType::Struct {
            type_args: expected_targs,
            ..
        } = expected_ty
        else {
            return None;
        };
        if expected_targs.is_empty() {
            return None;
        }

        let (&struct_name, module_path) = acc.path.split_last().unwrap_or_else(|| {
            panic!(
                "try_monomorphize_assoc_call: empty module path in static call `.{}` at {span}",
                self.context.resolve_string(&acc.member)
            )
        });

        let target_key = if self.ctx.structs.borrow().contains_key(&struct_name) {
            struct_name
        } else {
            let resolved = self
                .ctx
                .resolve_type_path_name(module_path, struct_name, *span);
            if !self.ctx.structs.borrow().contains_key(&resolved) {
                panic!(
                    "[try_monomorphize_assoc_call] resolve_type_path_name resolved bare name `{}` \
                     to `{}` (using ctx.module_idx = {}) at {span}, but no struct is registered \
                     under that key at all.",
                    struct_name, resolved, self.ctx.module_idx,
                );
            }
            resolved
        };

        let struct_methods_binding = self.ctx.struct_methods.borrow();
        let method_map = struct_methods_binding.get(&target_key).unwrap_or_else(|| {
            panic!(
                "[try_monomorphize_assoc_call] no methods registered for struct key `{}` \
                 (resolved from bare name `{}` in static call `.{}` at {span}). This almost \
                 always means target_key resolution at the call site doesn't match the key \
                 used when the `impl` block for this struct was registered. Known struct_methods keys: {:?}",
                target_key,
                struct_name,
                acc.member,
                struct_methods_binding
                    .keys()
                    .map(|k| self.context.resolve_string(k).to_string())
                    .collect::<Vec<_>>(),
            )
        });
        let base_method_name = *method_map.get(&acc.member).unwrap_or_else(|| {
            panic!(
                "try_monomorphize_assoc_call: struct `{}` has no method `{}` (static call at \
                 {span}). Known methods on this struct: {:?}",
                self.context.resolve_string(&target_key),
                self.context.resolve_string(&acc.member),
                method_map
                    .keys()
                    .map(|k| self.context.resolve_string(k).to_string())
                    .collect::<Vec<_>>(),
            )
        });
        drop(struct_methods_binding);

        let functions_binding = self.functions.borrow();
        let base_func = functions_binding
            .get(&base_method_name)
            .unwrap_or_else(|| {
                panic!(
                    "try_monomorphize_assoc_call: method `{}` resolved to function `{}` but \
                     that name isn't registered in the function table",
                    acc.member, base_method_name,
                )
            })
            .clone();
        drop(functions_binding);

        let struct_generics = self
            .ctx
            .structs
            .borrow()
            .get(&target_key)
            .unwrap_or_else(|| {
                panic!(
                    "try_monomorphize_assoc_call: struct key `{}` has a registered method `{}` \
                     but no struct declaration exists for it",
                    target_key, acc.member,
                )
            })
            .generics
            .clone();

        let type_params = base_func
            .generics
            .as_ref()
            .or(struct_generics.as_ref())
            .unwrap_or_else(|| {
                panic!(
                    "[try_monomorphize_assoc_call] call site `{}.{}(...)` at {span} expects {} \
                     concrete type argument(s) (target type: {:?}), but neither method `{}` nor \
                     struct `{}` declares any generic parameters",
                    struct_name,
                    acc.member,
                    expected_targs.len(),
                    expected_ty,
                    acc.member,
                    target_key,
                )
            });

        if type_params.len() != expected_targs.len() {
            panic!(
                "[try_monomorphize_assoc_call] `{}.{}` declares {} generic parameter(s) but the \
                 call site at {span} supplies {} type argument(s) via its expected type {:?}",
                struct_name,
                acc.member,
                type_params.len(),
                expected_targs.len(),
                expected_ty,
            );
        }

        let mut inner_subs: FxHashMap<StrId, HirType> = FxHashMap::default();
        for (p, a) in type_params.iter().zip(expected_targs.iter()) {
            let resolved = substitute_type(a, outer_subs, self.bump.clone());
            if contains_unresolved_generic(&resolved) {
                panic!(
                    "try_monomorphize_assoc_call: type argument `{:?}` for parameter `{}` in \
                     `{}.{}` at {span} still contains an unresolved generic after substitution \
                     against outer scope {:?}",
                    resolved,
                    p.name,
                    struct_name,
                    acc.member,
                    outer_subs.keys().collect::<Vec<_>>(),
                );
            }
            inner_subs.insert(p.name, resolved);
        }

        let new_args: Vec<HirExpr> = args
            .iter()
            .map(|a| self.monomorphize_expr(a, outer_subs))
            .collect();
        let args_slice = self.bump.alloc_slice(&new_args);

        let concrete_recv_ty = instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            target_key,
            expected_targs,
            self.bump.clone(),
        )
        .unwrap_or_else(|| {
            panic!(
                "[try_monomorphize_assoc_call] failed to instantiate `{}` with type args {:?} \
                 for call `{}.{}` at {span}, instantiate_struct_for_types returned None even \
                 though every arg was confirmed concrete above",
                target_key, expected_targs, struct_name, acc.member,
            )
        });
        let field_types: Vec<HirType> = concrete_recv_ty
            .fields
            .iter()
            .map(|f| f.field_type)
            .collect();
        let concrete_recv_ty = HirType::Struct {
            name: concrete_recv_ty.name,
            field_types: self.bump.alloc_slice_immutable(&field_types),
            type_args: &[],
        };

        let prev_self = self.current_this.replace(Some(concrete_recv_ty));
        let new_name = self
            .monomorphize_function(&base_func, &inner_subs)
            .unwrap_or_else(|| {
                panic!(
                    "try_monomorphize_assoc_call: monomorphize_function returned None for `{}.{}` \
                     at {span}",
                    struct_name, acc.member,
                )
            });
        self.current_this.replace(prev_self);

        Some(HirExpr::Call {
            callee: self
                .bump
                .alloc_value_immutable(HirExpr::Ident(new_name, acc.span)),
            args: args_slice,
            type_args: None,
            span: *span,
        })
    }

    fn instantiate_struct_ty_if_needed(&self, ty: HirType<'a, 'bump>) -> HirType<'a, 'bump> {
        let HirType::Struct {
            name, type_args, ..
        } = &ty
        else {
            return ty;
        };
        if type_args.is_empty() {
            return ty;
        }
        let Some(new_struct) = instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            *name,
            type_args,
            self.bump.clone(),
        ) else {
            return ty;
        };

        let field_types: Vec<HirType> = new_struct.fields.iter().map(|f| f.field_type).collect();
        HirType::Struct {
            name: new_struct.name,
            field_types: self.bump.alloc_slice_immutable(&field_types),
            type_args: &[],
        }
    }

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
                        .map(|t| substitute_type(t, subs, self.bump.clone()))
                        .collect();

                    let base_variants = self.ctx.enums.borrow().get(enum_name).map(|e| e.variants);

                    if let Some(_) = base_variants {
                        if let Some(new_enum) = instantiate_enum_for_types(
                            self.ctx,
                            &self.instantiated_enums,
                            &self.instantiated_enum_origins,
                            *enum_name,
                            &resolved_targs,
                            self.bump.clone(),
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
                    self.instantiate_type_recursively(substituted)
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
                    type_args: self.bump.alloc_slice_immutable(&new_type_args),
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
                if let (HirExpr::Ident(func_name, ident_span), Some(targs)) = (&**callee, type_args)
                {
                    let maybe_func = self.functions.borrow().get(func_name).cloned();
                    if let Some(func) = maybe_func {
                        if let Some(type_params) = func.generics {
                            if !type_params.is_empty() {
                                let resolved_targs: Vec<HirType> = targs
                                    .iter()
                                    .map(|t| substitute_type(t, subs, self.bump.clone()))
                                    .collect();
                                let mut inner_subs: FxHashMap<StrId, HirType> =
                                    FxHashMap::default();
                                type_params
                                    .iter()
                                    .zip(resolved_targs.iter())
                                    .for_each(|(p, a)| {
                                        inner_subs.insert(p.name, a.clone());
                                    });

                                let new_args: Vec<HirExpr> = args
                                    .iter()
                                    .map(|a| self.monomorphize_expr(a, subs))
                                    .collect();
                                let args_slice = self.bump.alloc_slice(&new_args);

                                if let Some(new_name) =
                                    self.monomorphize_function(&func, &inner_subs)
                                {
                                    let new_callee = HirExpr::Ident(new_name, *ident_span);
                                    return HirExpr::Call {
                                        callee: self.bump.alloc_value_immutable(new_callee),
                                        args: args_slice,
                                        type_args: None,
                                        span: *span,
                                    };
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
                        if matches!(concrete_recv_ty, HirType::Struct { .. }) {
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
                            .map(|t| substitute_type(t, subs, self.bump.clone()))
                            .collect();
                        &*self.bump.alloc_slice_immutable(&subd)
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

                        let resolved_targs: Vec<HirType> = targs
                            .iter()
                            .map(|t| substitute_type(t, subs, self.bump.clone()))
                            .collect();

                        let struct_ty = HirType::Struct {
                            name: target_key,
                            field_types: &[],
                            type_args: self.bump.alloc_slice_immutable(&resolved_targs),
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
                        .map(|t| substitute_type(t, subs, self.bump.clone()))
                        .collect();

                    if let Some(new_struct) = instantiate_struct_for_types(
                        self.ctx,
                        &self.instantiated_structs,
                        &self.instantiated_struct_origins,
                        &self.instantiated_enums,
                        &self.instantiated_enum_origins,
                        *struct_name,
                        &resolved_targs,
                        self.bump.clone(),
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
                mutable,
                span,
            } => {
                let new_expr = self.monomorphize_expr(expr, subs);
                HirExpr::Ref {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    mutable: *mutable,
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
            _ => expr.clone(),
        }
    }

    fn assert_fully_monomorphized(&self, items: &[Hir<'a, 'bump>]) {
        for item in items {
            match item {
                Hir::Func(f) => {
                    if let Some(body) = f.body {
                        self.assert_stmt_monomorphized(&body, f.name);
                    }
                }
                Hir::Impl(i) => {
                    if let Some(methods) = i.methods {
                        for m in methods.iter() {
                            if let Some(body) = m.body {
                                self.assert_stmt_monomorphized(&body, m.name);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn assert_stmt_monomorphized(&self, stmt: &HirStmt<'a, 'bump>, owner: StrId) {
        match stmt {
            HirStmt::Block { body } => {
                for s in body.iter() {
                    self.assert_stmt_monomorphized(s, owner);
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
            } => {
                self.assert_expr_monomorphized(cond, owner);
                for s in then_block.iter() {
                    self.assert_stmt_monomorphized(s, owner);
                }
                if let Some(e) = else_block {
                    self.assert_stmt_monomorphized(e, owner);
                }
            }
            HirStmt::While { cond, body } => {
                self.assert_expr_monomorphized(cond, owner);
                self.assert_stmt_monomorphized(body, owner);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.assert_stmt_monomorphized(i, owner);
                }
                if let Some(c) = condition {
                    self.assert_expr_monomorphized(c, owner);
                }
                if let Some(i) = increment {
                    self.assert_expr_monomorphized(i, owner);
                }
                self.assert_stmt_monomorphized(body, owner);
            }
            HirStmt::Let { value, .. } => self.assert_expr_monomorphized(value, owner),
            HirStmt::Return(Some(e)) => self.assert_expr_monomorphized(e, owner),
            HirStmt::Expr(e) => self.assert_expr_monomorphized(e, owner),
            HirStmt::UnsafeBlock { body } => self.assert_stmt_monomorphized(body, owner),
            HirStmt::Match { expr, arms } => {
                self.assert_expr_monomorphized(expr, owner);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.assert_expr_monomorphized(g, owner);
                    }
                    self.assert_stmt_monomorphized(arm.body, owner);
                }
            }
            HirStmt::Defer(s) => self.assert_stmt_monomorphized(s, owner),
            _ => {}
        }
    }

    fn assert_expr_monomorphized(&self, expr: &HirExpr<'a, 'bump>, owner: StrId) {
        match expr {
            HirExpr::Call {
                callee,
                type_args,
                span,
                args,
            } => {
                if let HirExpr::ModuleAccess(acc) = &**callee {
                    let (&struct_name, module_path) =
                        acc.path.split_last().unwrap_or((&acc.member, &[]));
                    let target_key = if self.ctx.structs.borrow().contains_key(&struct_name) {
                        struct_name
                    } else {
                        self.ctx
                            .resolve_type_path_name(module_path, struct_name, *span)
                    };
                    let still_generic = self
                        .ctx
                        .structs
                        .borrow()
                        .get(&target_key)
                        .map(|s| s.generics.is_some())
                        .unwrap_or(false);
                    if still_generic && type_args.is_none() {
                        panic!(
                            "[assert_fully_monomorphized] function `{}` still contains an \
                                 unresolved generic static call `.{}` on struct `{}` at {span}.",
                            owner, acc.member, target_key,
                        );
                    }
                }
                self.assert_expr_monomorphized(callee, owner);
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }

            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
            } => {
                let still_generic = self
                    .ctx
                    .enums
                    .borrow()
                    .get(enum_name)
                    .map(|e| e.generics.is_some())
                    .unwrap_or(false);
                if still_generic && type_args.is_none() {
                    panic!(
                        "[assert_fully_monomorphized] function `{}` still contains an unresolved \
                         generic enum construction `{}.{}` at {span}.",
                        owner, enum_name, variant,
                    );
                }
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }

            HirExpr::Match { expr, arms, .. } => {
                self.assert_expr_monomorphized(expr, owner);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.assert_expr_monomorphized(g, owner);
                    }
                    self.assert_stmt_monomorphized(arm.body, owner);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.assert_stmt_monomorphized(s, owner);
                }
            }
            HirExpr::If { if_stmt, .. } => self.assert_stmt_monomorphized(if_stmt, owner),
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.assert_expr_monomorphized(left, owner);
                self.assert_expr_monomorphized(right, owner);
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.assert_expr_monomorphized(object, owner);
            }
            HirExpr::Assignment { target, value, .. } => {
                self.assert_expr_monomorphized(target, owner);
                self.assert_expr_monomorphized(value, owner);
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.assert_expr_monomorphized(&f.value, owner);
                }
            }
            HirExpr::Cast { expr, .. }
            | HirExpr::Deref { expr, .. }
            | HirExpr::Ref { expr, .. } => self.assert_expr_monomorphized(expr, owner),
            HirExpr::Index { object, index, .. } => {
                self.assert_expr_monomorphized(object, owner);
                self.assert_expr_monomorphized(index, owner);
            }
            HirExpr::ArrayLiteral { elements, .. } => {
                for e in elements.iter() {
                    self.assert_expr_monomorphized(e, owner);
                }
            }
            HirExpr::Tuple(exprs, _) => {
                for e in exprs.iter() {
                    self.assert_expr_monomorphized(e, owner);
                }
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.assert_expr_monomorphized(object, owner);
                self.assert_expr_monomorphized(start, owner);
                self.assert_expr_monomorphized(end, owner);
            }
            HirExpr::Range { start, end, .. } => {
                self.assert_expr_monomorphized(start, owner);
                self.assert_expr_monomorphized(end, owner);
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.assert_expr_monomorphized(e, owner);
                    }
                }
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }
            HirExpr::InterfaceCall { callee, args, .. } => {
                self.assert_expr_monomorphized(callee, owner);
                for a in args.iter() {
                    self.assert_expr_monomorphized(a, owner);
                }
            }
            HirExpr::ExprList { list, .. } => {
                for e in list.iter() {
                    self.assert_expr_monomorphized(e, owner);
                }
            }

            _ => {}
        }
    }

    fn monomorphize_expr_with_expected_type<'subs>(
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
                .unwrap_or_else(|| self.monomorphize_expr(expr, subs)),
        }
    }

    fn monomorphize_stmt_with_expected_type<'subs>(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        expected_ty: &HirType<'a, 'bump>,
        substitutions: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirStmt<'a, 'bump> {
        match stmt {
            HirStmt::Expr(e) => {
                let new_expr = self
                    .try_monomorphize_enum_init_with_expected_type(e, expected_ty, substitutions)
                    .unwrap_or_else(|| {
                        self.monomorphize_expr_with_expected_type(e, expected_ty, substitutions)
                    });
                HirStmt::Expr(self.bump.alloc_value_immutable(new_expr))
            }
            HirStmt::Block { body } => {
                let Some((last, rest)) = body.split_last() else {
                    return HirStmt::Block { body };
                };
                let mut new_body: Vec<HirStmt> = rest
                    .iter()
                    .map(|s| self.monomorphize_stmt(s, substitutions))
                    .collect();
                new_body.push(self.monomorphize_stmt_with_expected_type(
                    last,
                    expected_ty,
                    substitutions,
                ));
                HirStmt::Block {
                    body: self.bump.alloc_slice(&new_body),
                }
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
            } => {
                let new_cond = self.monomorphize_expr(cond, substitutions);
                let new_then: Vec<HirStmt> = match then_block.split_last() {
                    Some((last, rest)) => {
                        let mut v: Vec<HirStmt> = rest
                            .iter()
                            .map(|s| self.monomorphize_stmt(s, substitutions))
                            .collect();
                        v.push(self.monomorphize_stmt_with_expected_type(
                            last,
                            expected_ty,
                            substitutions,
                        ));
                        v
                    }
                    None => Vec::new(),
                };
                let new_else = else_block.map(|e| {
                    let new_stmt =
                        self.monomorphize_stmt_with_expected_type(e, expected_ty, substitutions);
                    self.bump.alloc_value_immutable(new_stmt)
                });
                HirStmt::If {
                    cond: *self.bump.alloc_value_immutable(new_cond),
                    then_block: self.bump.alloc_slice(&new_then),
                    else_block: new_else,
                }
            }
            HirStmt::Match { expr, arms } => {
                let new_expr = self.monomorphize_expr(expr, substitutions);
                let new_arms: Vec<HirMatchArm> = arms
                    .iter()
                    .map(|arm| HirMatchArm {
                        pattern: arm.pattern.clone(),
                        guard: arm.guard.map(|g| {
                            self.bump
                                .alloc_value_immutable(self.monomorphize_expr(g, substitutions))
                        }),
                        body: self.bump.alloc_value_immutable(
                            self.monomorphize_stmt_with_expected_type(
                                arm.body,
                                expected_ty,
                                substitutions,
                            ),
                        ),
                    })
                    .collect();
                HirStmt::Match {
                    expr: self.bump.alloc_value_immutable(new_expr),
                    arms: self.bump.alloc_slice(&new_arms),
                }
            }
            HirStmt::UnsafeBlock { body } => {
                let new_body =
                    self.monomorphize_stmt_with_expected_type(body, expected_ty, substitutions);
                HirStmt::UnsafeBlock {
                    body: self.bump.alloc_value_immutable(new_body),
                }
            }
            other => self.monomorphize_stmt(other, substitutions),
        }
    }
}

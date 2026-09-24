use ir::{
    hir::{HirExpr, HirFunc, HirParam, HirType, StrId},
    ir_hasher::FxHashMap,
    span::SourceSpan,
};

use crate::hir_lowerer::monomorphization::{
    Monomorphizer,
    assertions::{contains_unresolved_generic, peel_to_struct},
    instantiate_struct_for_types,
    struct_instantiation::instantiate_enum_for_types,
    substitute_type,
};

impl<'a, 'bump, 'ctx> Monomorphizer<'a, 'bump, 'ctx> {
    pub(crate) fn substitute_type_with_self(
        &self,
        ty: &HirType<'a, 'bump>,
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        if let HirType::This = ty {
            if let Some(self_ty) = *self.current_this.borrow() {
                return self_ty;
            }
        }
        substitute_type(ty, subs, &self.bump)
    }

    pub(crate) fn instantiate_type_recursively(
        &self,
        ty: HirType<'a, 'bump>,
        span: SourceSpan<'a>,
    ) -> HirType<'a, 'bump> {
        let ty = self.instantiate_struct_ty_if_needed(ty, span);
        let ty = self.instantiate_enum_ty_if_needed(ty);
        match ty {
            HirType::Ref {
                inner,
                ref_kind: mutability_state,
                provenance,
            } => HirType::Ref {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                ref_kind: mutability_state,
                provenance,
            },
            HirType::SafePointer {
                inner,
                mutability_state,
            } => HirType::SafePointer {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                mutability_state,
            },
            HirType::UnsafePointer {
                inner,
                mutability_state,
            } => HirType::UnsafePointer {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                mutability_state,
            },
            HirType::OwnedPointer { inner, allocator } => HirType::OwnedPointer {
                inner: self
                    .bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                allocator,
            },
            HirType::Slice(inner) => HirType::Slice(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
            ),
            HirType::Array(inner, len) => HirType::Array(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
                len,
            ),
            HirType::Nullable(inner) => HirType::Nullable(
                self.bump
                    .alloc_value_immutable(self.instantiate_type_recursively(*inner, span)),
            ),
            HirType::Tuple(elems) => {
                let new_elems: Vec<HirType> = elems
                    .iter()
                    .map(|e| self.instantiate_type_recursively(*e, span))
                    .collect();
                HirType::Tuple(self.bump.alloc_slice(&new_elems))
            }
            other => other,
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
        let HirType::Struct {
            name, type_args, ..
        } = &ty
        else {
            return ty;
        };
        if type_args.is_empty() {
            return ty;
        }
        match instantiate_struct_for_types(
            self.ctx,
            &self.instantiated_structs,
            &self.instantiated_struct_origins,
            &self.instantiated_enums,
            &self.instantiated_enum_origins,
            *name,
            type_args,
            &self.bump,
        ) {
            Some(new_struct) => {
                let field_types: Vec<HirType> =
                    new_struct.fields.iter().map(|f| f.field_type).collect();
                HirType::Struct {
                    name: new_struct.name,
                    field_types: self.bump.alloc_slice(&field_types),
                    type_args: &[],
                }
            }
            None => panic!(
                "instantiate_struct_ty_if_needed: failed to instantiate `{}` with type args {:?} at {span}; \
                 this struct type would otherwise be silently left un-renamed, which produces \
                 confusing 'unknown struct' failures much later in MIR lowering. Please open an issue if you see this error message.",
                name, type_args
            ),
        }
    }

    pub(crate) fn resolve_method_for_type(
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
            let origin: Option<(StrId, Vec<HirType<'a, 'bump>>)> =
                self.instantiated_struct_origins.borrow().get(name).cloned();

            if let Some((origin_name, origin_targs)) = origin {
                let targs_slice = self.bump.alloc_slice(&origin_targs);
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
            *name,
            type_args,
            &self.bump,
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
            field_types: self.bump.alloc_slice(&field_types),
            type_args: &[],
        };

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

    pub(crate) fn field_type_of(
        &self,
        ty: &HirType<'a, 'bump>,
        field: StrId,
    ) -> Option<HirType<'a, 'bump>> {
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

    pub(crate) fn instantiate_enum_ty_if_needed(
        &self,
        ty: HirType<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
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
            &self.bump,
        ) else {
            return ty;
        };
        HirType::Enum {
            name: new_enum.name,
            type_args: &[],
            variants: new_enum.variants,
        }
    }
}

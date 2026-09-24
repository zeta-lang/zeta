use ir::{
    hir::{HirParam, HirType, StrId},
    ir_hasher::FxHashMap,
};

use crate::{str_id_to_string, TypeChecker};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn instantiate_struct(
        &self,
        name: StrId,
        args: &[HirType<'a, 'bump>],
    ) -> Option<&'bump [HirType<'a, 'bump>]> {
        if let Some(cached) = self.context.get_struct_instantiation(name, args) {
            return Some(cached);
        }

        let name_str = str_id_to_string(name);
        let def = self.context.get_struct(&name_str)?;
        let generics = def.generics?;
        let mut full_args = args.to_vec();
        if full_args.len() < generics.len() {
            let mut subs = FxHashMap::default();
            for (param, arg) in generics.iter().zip(full_args.iter()) {
                subs.insert(param.name, *arg);
            }
            let start = full_args.len();
            for param in &generics[start..] {
                if let Some(ref def_ty) = param.default_type {
                    let resolved = self.substitute_type_local(def_ty, &subs);
                    subs.insert(param.name, resolved);
                    full_args.push(resolved);
                } else {
                    return None;
                }
            }
        }
        if generics.len() != full_args.len() {
            return None;
        }

        let mut subs = FxHashMap::default();
        for (param, arg) in generics.iter().zip(full_args.iter()) {
            subs.insert(param.name, *arg);
        }

        let field_types: Vec<_> = def
            .fields
            .iter()
            .map(|f| self.substitute_type_local(&f.field_type, &subs))
            .collect();
        let result = self.context.bump.alloc_slice_copy(&field_types);

        self.context
            .cache_struct_instantiation(name, args.to_vec(), result);
        Some(result)
    }

    pub fn instantiate_enum(
        &self,
        name: StrId,
        args: &[HirType<'a, 'bump>],
    ) -> Option<&'bump [(StrId, &'bump [HirType<'a, 'bump>])]> {
        if let Some(cached) = self.context.get_enum_instantiation(name, args) {
            return Some(cached);
        }

        let name_str = str_id_to_string(name);
        let def = self.context.get_enum(&name_str)?;
        let generics = def.generics?;
        if generics.len() != args.len() {
            return None;
        }

        let mut subs = FxHashMap::default();
        for (param, arg) in generics.iter().zip(args.iter()) {
            subs.insert(param.name, *arg);
        }

        let mut resolved_variants = Vec::with_capacity(def.variants.len());
        for variant in def.variants.iter() {
            let field_types: Vec<_> = variant
                .fields
                .iter()
                .map(|f| self.substitute_type_local(&f.field_type, &subs))
                .collect();
            resolved_variants.push((
                variant.name,
                self.context.bump.alloc_slice_copy(&field_types),
            ));
        }
        let result = self.context.bump.alloc_slice_copy(&resolved_variants);

        self.context
            .cache_enum_instantiation(name, args.to_vec(), result);
        Some(result)
    }

    pub fn substitute_type_local(
        &self,
        ty: &HirType<'a, 'bump>,
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        match ty {
            HirType::Generic(name) => subs.get(name).copied().unwrap_or(*ty),
            HirType::Nullable(inner) => HirType::Nullable(
                self.context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
            ),
            HirType::Array(inner, len) => HirType::Array(
                self.context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                *len,
            ),
            HirType::Slice(inner) => HirType::Slice(
                self.context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
            ),
            HirType::SafePointer {
                inner,
                mutability_state,
            } => HirType::SafePointer {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                mutability_state: *mutability_state,
            },
            HirType::UnsafePointer {
                inner,
                mutability_state,
            } => HirType::UnsafePointer {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                mutability_state: *mutability_state,
            },
            HirType::OwnedPointer { inner, allocator } => HirType::OwnedPointer {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                allocator: *allocator,
            },
            HirType::Ref {
                inner,
                ref_kind: mutability_state,
                provenance,
            } => HirType::Ref {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                ref_kind: *mutability_state,
                provenance: *provenance,
            },
            HirType::Tuple(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.substitute_type_local(e, subs))
                    .collect();
                HirType::Tuple(self.context.bump.alloc_slice_copy(&new_elems))
            }

            HirType::Struct {
                name,
                field_types,
                type_args,
            } => {
                let new_fields: Vec<_> = field_types
                    .iter()
                    .map(|f| self.substitute_type_local(f, subs))
                    .collect();
                let new_args: Vec<_> = type_args
                    .iter()
                    .map(|a| self.substitute_type_local(a, subs))
                    .collect();
                HirType::Struct {
                    name: *name,
                    field_types: self.context.bump.alloc_slice_copy(&new_fields),
                    type_args: self.context.bump.alloc_slice_copy(&new_args),
                }
            }
            HirType::Enum {
                name,
                type_args,
                variants,
            } => {
                let new_args: Vec<_> = type_args
                    .iter()
                    .map(|a| self.substitute_type_local(a, subs))
                    .collect();
                HirType::Enum {
                    name: *name,
                    type_args: self.context.bump.alloc_slice_copy(&new_args),
                    variants,
                }
            }
            HirType::Dyn { bounds } => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| self.substitute_type_local(b, subs))
                    .collect();
                HirType::Dyn {
                    bounds: self.context.bump.alloc_slice_copy(&new_bounds),
                }
            }
            HirType::Lambda {
                params,
                return_type,
            } => {
                let new_params: Vec<_> = params
                    .iter()
                    .map(|p| self.substitute_type_local(p, subs))
                    .collect();
                HirType::Lambda {
                    params: self.context.bump.alloc_slice_copy(&new_params),
                    return_type: self
                        .context
                        .bump
                        .alloc_value(self.substitute_type_local(return_type, subs)),
                }
            }

            _ => *ty,
        }
    }

    pub fn substitute_params_local(
        &self,
        params: &[HirParam<'a, 'bump>],
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> &'bump [HirParam<'a, 'bump>] {
        let new_params: Vec<HirParam> = params
            .iter()
            .map(|p| match p {
                HirParam::Normal {
                    name,
                    param_type,
                    multi_place,
                    span,
                } => HirParam::Normal {
                    name: *name,
                    param_type: self.substitute_type_local(param_type, subs),
                    multi_place: *multi_place,
                    span: *span,
                },
                HirParam::This {
                    kind,
                    span,
                    multi_place,
                } => HirParam::This {
                    kind: *kind,
                    span: *span,
                    multi_place: *multi_place,
                },
            })
            .collect();
        self.context.bump.alloc_slice(&new_params)
    }

    pub fn unify_generic(
        &self,
        declared: &HirType<'a, 'bump>,
        actual: &HirType<'a, 'bump>,
        subs: &mut FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        match declared {
            HirType::Generic(name) => {
                subs.entry(*name).or_insert(*actual);
            }
            HirType::Nullable(inner) => {
                if let HirType::Nullable(actual_inner) = actual {
                    self.unify_generic(inner, actual_inner, subs);
                }
            }
            HirType::Array(inner, _) => {
                if let HirType::Array(actual_inner, _) = actual {
                    self.unify_generic(inner, actual_inner, subs);
                }
            }
            HirType::Slice(inner) => {
                if let HirType::Slice(actual_inner) = actual {
                    self.unify_generic(inner, actual_inner, subs);
                }
            }
            HirType::Ref { inner, .. } => {
                self.unify_generic(inner, Self::strip_ref(actual), subs);
            }
            _ => {}
        }
    }

    pub fn generic_substitutions_for_struct(
        &self,
        struct_name: StrId,
        type_args: &[HirType<'a, 'bump>],
    ) -> FxHashMap<StrId, HirType<'a, 'bump>> {
        let mut subs = FxHashMap::default();
        if type_args.is_empty() {
            return subs;
        }
        let name_str = str_id_to_string(struct_name);
        if let Some(def) = self.context.get_struct(&name_str) {
            if let Some(generics) = def.generics {
                for (param, arg) in generics.iter().zip(type_args.iter()) {
                    subs.insert(param.name, *arg);
                }
            }
        }
        subs
    }
}

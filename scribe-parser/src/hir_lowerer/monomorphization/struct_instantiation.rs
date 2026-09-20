use super::naming::instantiate_struct_name;
use super::type_substitution::substitute_type;
use crate::hir_lowerer::context::LoweringCtx;
use crate::hir_lowerer::monomorphization::monomorphizer::contains_unresolved_generic;
use crate::hir_lowerer::monomorphization::naming::instantiate_enum_name;
use crate::hir_lowerer::monomorphization::suffix_for_subs;
use ir::hir::{HirEnum, HirEnumVariant, HirField, HirStruct, HirType, StrId};
use ir::ir_hasher::FxHashMap;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;
use zetaruntime::arena::GrowableAtomicBump;

pub fn instantiate_enum_for_types<'a, 'bump>(
    ctx: &LoweringCtx<'a, 'bump>,
    instantiated_enums: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_enum_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    base_id: StrId,
    concrete_args: &[HirType<'a, 'bump>],
    bump: Arc<GrowableAtomicBump<'bump>>,
) -> Option<&'bump HirEnum<'a, 'bump>> {
    if concrete_args.is_empty() {
        return ctx
            .enums
            .borrow()
            .get(&base_id)
            .map(|c| unsafe { std::mem::transmute::<_, &'bump _>(c) });
    }

    if concrete_args
        .iter()
        .any(|t| matches!(t, HirType::Generic(_)))
    {
        return None;
    }

    let binding = ctx.enums.borrow();
    let base = binding.get(&base_id)?;
    let type_map = if let Some(generics) = &base.generics {
        let mut map = FxHashMap::default();
        for (param, arg) in generics.iter().zip(concrete_args) {
            map.insert(param.name, arg.clone());
        }
        map
    } else {
        FxHashMap::default()
    };
    let suffix = suffix_for_subs(ctx.context.clone(), &type_map);
    let key = (base_id, suffix);

    {
        let cache = instantiated_enums.borrow();
        if let Some(cached) = cache.get(&key) {
            let ptr = ctx
                .enums
                .borrow()
                .get(cached)
                .map(|c| unsafe { std::mem::transmute::<_, &'bump _>(c) });
            if ptr.is_some() {
                return ptr;
            }
        }
    }
    drop(binding);

    let base = {
        let enums = ctx.enums.borrow();
        enums.get(&base_id)?.clone()
    };
    let mut new_enum = base.clone();
    let interned = instantiate_enum_name(concrete_args, &base, ctx.context.clone());
    new_enum.name = interned;

    if let Some(generics) = &base.generics {
        let mut type_map = FxHashMap::default();
        for (param, arg) in generics.iter().zip(concrete_args) {
            type_map.insert(param.name, arg.clone());
        }
        let mut new_variants: Vec<HirEnumVariant> = Vec::with_capacity(base.variants.len());
        for variant in new_enum.variants {
            let mut new_fields: Vec<HirField> = Vec::with_capacity(variant.fields.len());
            for field in variant.fields {
                let new_field_type = substitute_type(&field.field_type, &type_map, bump.clone());
                new_fields.push(HirField {
                    name: field.name,
                    visibility: field.visibility,
                    field_type: new_field_type,
                });
            }
            new_variants.push(HirEnumVariant {
                name: variant.name,
                fields: bump.alloc_slice(&new_fields),
            });
        }
        new_enum.variants = bump.alloc_slice(&new_variants);
        new_enum.generics = None;
    }

    let new_enum_ptr = bump.alloc_value(new_enum);
    {
        let mut enums = ctx.enums.borrow_mut();
        enums.insert(interned, *new_enum_ptr);
        instantiated_enums
            .borrow_mut()
            .insert(key, new_enum_ptr.name);
        instantiated_enum_origins
            .borrow_mut()
            .insert(interned, (base_id, concrete_args.to_vec()));
    }

    let interfaces = { ctx.struct_interfaces.borrow().get(&base_id).cloned() };
    if let Some(interfaces) = interfaces {
        ctx.struct_interfaces
            .borrow_mut()
            .insert(interned, interfaces);
    }
    let methods = { ctx.struct_methods.borrow().get(&base_id).cloned() };
    if let Some(methods) = methods {
        ctx.struct_methods.borrow_mut().insert(interned, methods);
    }

    Some(new_enum_ptr)
}

pub fn instantiate_struct_for_types<'a, 'bump>(
    ctx: &LoweringCtx<'a, 'bump>,
    instantiated_structs: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_struct_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    instantiated_enums: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_enum_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    base_id: StrId,
    concrete_args: &[HirType<'a, 'bump>],
    bump: Arc<GrowableAtomicBump<'bump>>,
) -> Option<&'bump HirStruct<'a, 'bump>> {
    if concrete_args.is_empty() {
        return ctx
            .structs
            .borrow()
            .get(&base_id)
            .map(|c| unsafe { std::mem::transmute::<_, &'bump _>(c) });
    }

    if concrete_args
        .iter()
        .any(|t| matches!(t, HirType::Generic(_)))
    {
        return None;
    }

    let binding = ctx.structs.borrow();
    let base = binding.get(&base_id)?;
    let mut full_concrete_args = concrete_args.to_vec();
    if let Some(generics) = &base.generics {
        if full_concrete_args.len() < generics.len() {
            let mut subs = FxHashMap::default();
            for (param, arg) in generics.iter().zip(full_concrete_args.iter()) {
                subs.insert(param.name, arg.clone());
            }
            let start = full_concrete_args.len();
            for param in &generics[start..] {
                if let Some(ref def_ty) = param.default_type {
                    let resolved = substitute_type(def_ty, &subs, bump.clone());
                    subs.insert(param.name, resolved.clone());
                    full_concrete_args.push(resolved);
                } else {
                    return None;
                }
            }
        }
    }
    let type_map = if let Some(generics) = &base.generics {
        let mut map = FxHashMap::default();
        for (param, arg) in generics.iter().zip(full_concrete_args.iter()) {
            map.insert(param.name, arg.clone());
        }
        map
    } else {
        FxHashMap::default()
    };
    let suffix = suffix_for_subs(ctx.context.clone(), &type_map);
    let key = (base_id, suffix);
    {
        let cache = instantiated_structs.borrow();
        if let Some(cached) = cache.get(&key) {
            let ptr = ctx
                .structs
                .borrow()
                .get(cached)
                .map(|c| unsafe { std::mem::transmute::<_, &'bump _>(c) });
            if ptr.is_some() {
                return ptr;
            }
        }
    }

    let interned = instantiate_struct_name(&full_concrete_args, base, ctx.context.clone());
    let base_owned = base.clone();
    drop(binding);

    let mut placeholder = base_owned.clone();
    placeholder.name = interned;
    placeholder.generics = None;
    ctx.structs.borrow_mut().insert(interned, placeholder);
    instantiated_structs.borrow_mut().insert(key, interned);
    instantiated_struct_origins
        .borrow_mut()
        .insert(interned, (base_id, full_concrete_args.clone()));

    let interfaces = { ctx.struct_interfaces.borrow().get(&base_id).cloned() };
    if let Some(interfaces) = interfaces {
        ctx.struct_interfaces
            .borrow_mut()
            .insert(interned, interfaces);
    }
    let methods = { ctx.struct_methods.borrow().get(&base_id).cloned() };
    if let Some(methods) = methods {
        ctx.struct_methods.borrow_mut().insert(interned, methods);
    }

    let mut new_struct = base_owned.clone();
    new_struct.name = interned;
    if let Some(generics) = &base_owned.generics {
        let mut type_map = FxHashMap::default();
        for (param, arg) in generics.iter().zip(full_concrete_args.iter()) {
            type_map.insert(param.name, arg.clone());
        }
        let mut new_fields = Vec::new();
        for field in new_struct.fields {
            let substituted = substitute_type(&field.field_type, &type_map, bump.clone());

            let new_field_type = instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                substituted,
                bump.clone(),
            );
            new_fields.push(HirField {
                name: field.name,
                visibility: field.visibility,
                field_type: new_field_type,
            });
        }
        new_struct.fields = bump.alloc_slice(&new_fields);
        new_struct.generics = None;
    }

    let new_struct_ptr = bump.alloc_value(new_struct.clone());
    ctx.structs.borrow_mut().insert(interned, new_struct);

    Some(new_struct_ptr)
}

pub fn direct_method_lookup<'a, 'bump>(
    ctx: &LoweringCtx<'a, 'bump>,
    instantiated_structs: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_struct_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    instantiated_enums: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_enum_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    struct_name: &StrId,
    struct_args: &[HirType<'a, 'bump>],
    method_name: &StrId,
    bump: Arc<GrowableAtomicBump<'bump>>,
) -> Option<StrId> {
    let concrete_name = if struct_args.is_empty() {
        *struct_name
    } else {
        instantiate_struct_for_types(
            ctx,
            instantiated_structs,
            instantiated_struct_origins,
            instantiated_enums,
            instantiated_enum_origins,
            *struct_name,
            struct_args,
            bump.clone(),
        )?
        .name
    };
    let struct_methods = ctx.struct_methods.borrow();
    let methods = struct_methods.get(&concrete_name)?;
    methods.get(method_name).map(|m| *m)
}

fn instantiate_type_recursively_ctx<'a, 'bump>(
    ctx: &LoweringCtx<'a, 'bump>,
    instantiated_structs: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_struct_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    instantiated_enums: &Rc<RefCell<FxHashMap<(StrId, StrId), StrId>>>,
    instantiated_enum_origins: &Rc<RefCell<FxHashMap<StrId, (StrId, Vec<HirType<'a, 'bump>>)>>>,
    ty: HirType<'a, 'bump>,
    bump: Arc<GrowableAtomicBump<'bump>>,
) -> HirType<'a, 'bump> {
    match ty {
        HirType::Struct {
            name, type_args, ..
        } => {
            if !type_args.is_empty() && !type_args.iter().any(|t| matches!(t, HirType::Generic(_)))
            {
                let rec_args: Vec<_> = type_args
                    .iter()
                    .map(|a| {
                        instantiate_type_recursively_ctx(
                            ctx,
                            instantiated_structs,
                            instantiated_struct_origins,
                            instantiated_enums,
                            instantiated_enum_origins,
                            *a,
                            bump.clone(),
                        )
                    })
                    .collect();
                if let Some(nested_inst) = instantiate_struct_for_types(
                    ctx,
                    instantiated_structs,
                    instantiated_struct_origins,
                    instantiated_enums,
                    instantiated_enum_origins,
                    name,
                    &rec_args,
                    bump.clone(),
                ) {
                    let still_generic = nested_inst
                        .fields
                        .iter()
                        .any(|f| contains_unresolved_generic(&f.field_type));
                    let field_types: &[HirType] = if still_generic {
                        &[]
                    } else {
                        let nested_fields: Vec<HirType> =
                            nested_inst.fields.iter().map(|f| f.field_type).collect();
                        bump.alloc_slice_immutable(&nested_fields)
                    };
                    HirType::Struct {
                        name: nested_inst.name,
                        field_types,
                        type_args: &[],
                    }
                } else {
                    ty
                }
            } else {
                ty
            }
        }
        HirType::Enum {
            name,
            type_args,
            variants: _,
        } => {
            if !type_args.is_empty() && !type_args.iter().any(|t| matches!(t, HirType::Generic(_)))
            {
                let rec_args: Vec<_> = type_args
                    .iter()
                    .map(|a| {
                        instantiate_type_recursively_ctx(
                            ctx,
                            instantiated_structs,
                            instantiated_struct_origins,
                            instantiated_enums,
                            instantiated_enum_origins,
                            *a,
                            bump.clone(),
                        )
                    })
                    .collect();
                if let Some(nested_inst) = instantiate_enum_for_types(
                    ctx,
                    instantiated_enums,
                    instantiated_enum_origins,
                    name,
                    &rec_args,
                    bump.clone(),
                ) {
                    HirType::Enum {
                        name: nested_inst.name,
                        variants: nested_inst.variants,
                        type_args: &[],
                    }
                } else {
                    ty
                }
            } else {
                ty
            }
        }
        HirType::Slice(inner) => {
            HirType::Slice(bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )))
        }
        HirType::Array(inner, len) => HirType::Array(
            bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )),
            len,
        ),
        HirType::Ref {
            inner,
            ref_kind,
            provenance,
        } => HirType::Ref {
            inner: bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )),
            ref_kind,
            provenance,
        },
        HirType::SafePointer {
            inner,
            mutability_state,
        } => HirType::SafePointer {
            inner: bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )),
            mutability_state,
        },
        HirType::UnsafePointer {
            inner,
            mutability_state,
        } => HirType::UnsafePointer {
            inner: bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )),
            mutability_state,
        },
        HirType::OwnedPointer { inner, allocator } => HirType::OwnedPointer {
            inner: bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )),
            allocator,
        },
        HirType::Nullable(inner) => {
            HirType::Nullable(bump.alloc_value_immutable(instantiate_type_recursively_ctx(
                ctx,
                instantiated_structs,
                instantiated_struct_origins,
                instantiated_enums,
                instantiated_enum_origins,
                *inner,
                bump.clone(),
            )))
        }
        HirType::Tuple(hir_types) => {
            let types: Vec<HirType<'a, 'bump>> = hir_types
                .iter()
                .map(|inner| {
                    instantiate_type_recursively_ctx(
                        ctx,
                        instantiated_structs,
                        instantiated_struct_origins,
                        instantiated_enums,
                        instantiated_enum_origins,
                        *inner,
                        bump.clone(),
                    )
                })
                .collect::<Vec<_>>();
            HirType::Tuple(bump.alloc_slice_immutable(types.as_slice()))
        }
        other => other,
    }
}

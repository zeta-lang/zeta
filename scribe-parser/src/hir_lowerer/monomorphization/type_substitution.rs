use ir::hir::{HirEnumVariant, HirField, HirType, HirTypeKind, StrId};
use ir::ir_hasher::HashMap;
use zetaruntime::bump::GrowableBump;

pub fn substitute_type<'a, 'subs, 'bump>(
    ty: &HirType<'a, 'bump>,
    subs: &'subs HashMap<StrId, HirType<'a, 'bump>>,
    bump: &'bump GrowableBump<'bump>,
) -> HirType<'a, 'bump> {
    match &ty.kind {
        HirTypeKind::Generic(name) => subs.get(name).copied().unwrap_or(*ty),
        HirTypeKind::Slice(inner) => HirType::new(
            HirTypeKind::Slice(bump.alloc_value_immutable(substitute_type(inner, subs, bump))),
            ty.span,
        ),
        HirTypeKind::Ref {
            inner,
            ref_kind,
            provenance,
        } => HirType::new(
            HirTypeKind::Ref {
                inner: bump.alloc_value_immutable(substitute_type(inner, subs, bump)),
                ref_kind: *ref_kind,
                provenance: *provenance,
            },
            ty.span,
        ),
        HirTypeKind::SafePointer {
            inner,
            mutability_state,
        } => HirType::new(
            HirTypeKind::SafePointer {
                inner: bump.alloc_value_immutable(substitute_type(inner, subs, bump)),
                mutability_state: *mutability_state,
            },
            ty.span,
        ),
        HirTypeKind::UnsafePointer {
            inner,
            mutability_state,
        } => HirType::new(
            HirTypeKind::UnsafePointer {
                inner: bump.alloc_value_immutable(substitute_type(inner, subs, bump)),
                mutability_state: *mutability_state,
            },
            ty.span,
        ),
        HirTypeKind::OwnedPointer { inner, allocator } => HirType::new(
            HirTypeKind::OwnedPointer {
                inner: bump.alloc_value_immutable(substitute_type(inner, subs, bump)),
                allocator: *allocator,
            },
            ty.span,
        ),
        HirTypeKind::Struct {
            name,
            unmangled_name,
            field_types,
            type_args,
        } => {
            let new_fields: Vec<HirType<'a, 'bump>> = field_types
                .iter()
                .map(|f| substitute_type(f, subs, bump))
                .collect();
            let new_args: Vec<_> = type_args
                .iter()
                .map(|a| substitute_type(a, subs, bump))
                .collect();
            HirType::new(
                HirTypeKind::Struct {
                    name: *name,
                    unmangled_name: *unmangled_name,
                    field_types: bump.alloc_slice(&new_fields),
                    type_args: bump.alloc_slice(&new_args),
                },
                ty.span,
            )
        }

        HirTypeKind::DynInterface {
            name,
            unmangled_name,
            type_args,
        } => {
            let new_args: Vec<HirType<'a, 'bump>> = type_args
                .iter()
                .map(|a| substitute_type(a, subs, bump))
                .collect();
            HirType::new(
                HirTypeKind::DynInterface {
                    name: *name,
                    unmangled_name: *unmangled_name,
                    type_args: bump.alloc_slice(&new_args),
                },
                ty.span,
            )
        }

        HirTypeKind::Enum {
            name,
            unmangled_name,
            variants,
            type_args,
        } => {
            let new_args: Vec<HirType<'a, 'bump>> = type_args
                .iter()
                .map(|a| substitute_type(a, subs, bump))
                .collect();

            let new_variants: Vec<HirEnumVariant<'a, 'bump>> = variants
                .iter()
                .map(|v| {
                    let new_fields: Vec<HirField<'a, 'bump>> = v
                        .fields
                        .iter()
                        .map(|f| HirField {
                            name: f.name,
                            visibility: f.visibility,
                            field_type: substitute_type(&f.field_type, subs, bump),
                            manual: f.manual,
                            span: f.span,
                        })
                        .collect();
                    HirEnumVariant {
                        name: v.name,
                        fields: bump.alloc_slice(&new_fields),
                        span: v.span,
                    }
                })
                .collect();

            HirType::new(
                HirTypeKind::Enum {
                    name: *name,
                    unmangled_name: *unmangled_name,
                    variants: bump.alloc_slice(&new_variants),
                    type_args: bump.alloc_slice(&new_args),
                },
                ty.span,
            )
        }

        HirTypeKind::Lambda {
            params,
            return_type,
        } => {
            let new_params: Vec<HirType<'a, 'bump>> = params
                .iter()
                .map(|p| substitute_type(p, subs, bump))
                .collect();
            let new_return = substitute_type(return_type, subs, bump);
            HirType::new(
                HirTypeKind::Lambda {
                    params: bump.alloc_slice(&new_params),
                    return_type: bump.alloc_value_immutable(new_return),
                },
                ty.span,
            )
        }
        HirTypeKind::Array(inner, len) => HirType::new(
            HirTypeKind::Array(
                bump.alloc_value_immutable(substitute_type(inner, subs, bump)),
                *len,
            ),
            ty.span,
        ),
        HirTypeKind::Nullable(inner) => HirType::new(
            HirTypeKind::Nullable(bump.alloc_value_immutable(substitute_type(inner, subs, bump))),
            ty.span,
        ),
        HirTypeKind::Tuple(elems) => {
            let new_elems: Vec<HirType<'a, 'bump>> = elems
                .iter()
                .map(|e| substitute_type(e, subs, bump))
                .collect();
            HirType::new(HirTypeKind::Tuple(bump.alloc_slice(&new_elems)), ty.span)
        }
        HirTypeKind::Dyn { bounds } => {
            let new_bounds: Vec<HirType<'a, 'bump>> = bounds
                .iter()
                .map(|b| substitute_type(b, subs, bump))
                .collect();
            HirType::new(
                HirTypeKind::Dyn {
                    bounds: bump.alloc_slice(&new_bounds),
                },
                ty.span,
            )
        }
        HirTypeKind::Range { elem, .. } => {
            let mut new_ty = *ty;
            if let HirTypeKind::Range { elem: e, .. } = &mut new_ty.kind {
                *e = bump.alloc_value_immutable(substitute_type(elem, subs, bump));
            }
            new_ty
        }
        _ => *ty,
    }
}

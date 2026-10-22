use crate::{
    hir::{AssignmentOperator, HirEnum, HirStruct, HirType, HirTypeKind, Operator, StrId},
    ir_hasher::HashMap,
    ssa_ir::{BinOp, SsaType},
};

pub fn assign_op_to_bin_op(op: AssignmentOperator) -> BinOp {
    let bin_op = match op {
        AssignmentOperator::AddAssign => BinOp::Add,
        AssignmentOperator::SubtractAssign => BinOp::Sub,
        AssignmentOperator::MultiplyAssign => BinOp::Mul,
        AssignmentOperator::DivideAssign => BinOp::Div,
        AssignmentOperator::ModuloAssign => BinOp::Mod,
        AssignmentOperator::BitAndAssign => BinOp::BitAnd,
        AssignmentOperator::BitOrAssign => BinOp::BitOr,
        AssignmentOperator::BitXorAssign => BinOp::BitXor,
        AssignmentOperator::ShiftLeftAssign => BinOp::ShiftLeft,
        AssignmentOperator::ShiftRightAssign => BinOp::ShiftRight,
        _ => unreachable!(),
    };
    bin_op
}

pub fn lower_type_hir(
    ty: &HirType,
    enums: &HashMap<StrId, HirEnum<'_, '_>>,
    structs: &HashMap<StrId, HirStruct<'_, '_>>,
) -> SsaType {
    lower_type_hir_inner(ty, enums, structs, &mut Vec::new())
}

fn lower_type_hir_inner(
    ty: &HirType,
    enums: &HashMap<StrId, HirEnum<'_, '_>>,
    structs: &HashMap<StrId, HirStruct<'_, '_>>,
    in_progress: &mut Vec<StrId>,
) -> SsaType {
    match ty.kind {
        HirTypeKind::I8 => SsaType::I8,
        HirTypeKind::I16 => SsaType::I16,
        HirTypeKind::I32 => SsaType::I32,
        HirTypeKind::I64 => SsaType::I64,
        HirTypeKind::I128 => SsaType::I128,
        HirTypeKind::U8 => SsaType::U8,
        HirTypeKind::U16 => SsaType::U16,
        HirTypeKind::U32 => SsaType::U32,
        HirTypeKind::U64 => SsaType::U64,
        HirTypeKind::U128 => SsaType::U64,
        HirTypeKind::F32 => SsaType::F32,
        HirTypeKind::F64 => SsaType::F64,
        HirTypeKind::Boolean => SsaType::Bool,
        HirTypeKind::String => SsaType::String,

        HirTypeKind::Struct {
            name, field_types, ..
        } => {
            if in_progress.contains(&name) {
                let unmangled = structs.get(&name).map(|s| s.unmangled_name).unwrap_or(name);
                return SsaType::User(name, unmangled, Vec::new());
            }

            match structs.get(&name).filter(|s| s.generics.is_none()) {
                Some(def) => {
                    in_progress.push(name);
                    let mut fields = Vec::with_capacity(def.fields.len());
                    for f in def.fields.iter() {
                        fields.push(lower_type_hir_inner(
                            &f.field_type,
                            enums,
                            structs,
                            in_progress,
                        ));
                    }
                    in_progress.pop();
                    SsaType::User(name, def.unmangled_name, fields)
                }
                // Not registered yet: fall back to what the type carries.
                None => {
                    let mut fields = Vec::with_capacity(field_types.len());
                    for t in field_types.iter() {
                        fields.push(lower_type_hir_inner(t, enums, structs, in_progress));
                    }
                    SsaType::User(name, name, fields)
                }
            }
        }

        HirTypeKind::DynInterface {
            name,
            unmangled_name,
            ..
        } => SsaType::Interface(name, unmangled_name),

        HirTypeKind::Enum {
            name,
            unmangled_name,
            ..
        } => {
            if in_progress.contains(&name) {
                // Behind a pointer; the layout never looks at the payload.
                return SsaType::Enum {
                    name: name,
                    unmangled_name: unmangled_name,
                    variants: Vec::new(),
                };
            }
            let hir_enum = enums.get(&name).unwrap_or_else(|| {
                println!("enums: {:?}", enums.keys());
                panic!(
                    "lower_type_hir: enum `{}` not found in registry, it must be \
                             registered (post-monomorphization, under its final concrete name) \
                             before any HirType::Enum referencing it is lowered",
                    name
                )
            });
            in_progress.push(name);
            let variants = hir_enum
                .variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|f| lower_type_hir_inner(&f.field_type, enums, structs, in_progress))
                        .collect()
                })
                .collect();
            in_progress.pop();
            SsaType::Enum {
                name,
                unmangled_name,
                variants,
            }
        }

        HirTypeKind::Void => SsaType::Void,
        HirTypeKind::SafePointer {
            inner,
            mutability_state,
        }
        | HirTypeKind::UnsafePointer {
            inner,
            mutability_state,
        } => {
            // `*dyn T` / `[*]dyn T`: don't drill into the interface's own
            // field layout, that's only meaningful for the implementor.
            // A pointer-to-dyn is vtable-dispatched, same shape regardless
            // of which concrete type is behind it.
            let ptr_kind = if mutability_state == crate::ast::MutabilityState::Const {
                if ty.is_safe_pointer() {
                    crate::ssa_ir::SsaPointerKind::SafeConst
                } else {
                    crate::ssa_ir::SsaPointerKind::UnsafeConst
                }
            } else {
                if ty.is_safe_pointer() {
                    crate::ssa_ir::SsaPointerKind::SafeMut
                } else {
                    crate::ssa_ir::SsaPointerKind::UnsafeMut
                }
            };
            match inner.kind {
                HirTypeKind::DynInterface {
                    name,
                    unmangled_name,
                    ..
                } => SsaType::Pointer(ptr_kind, Box::new(SsaType::Interface(name, unmangled_name))),
                HirTypeKind::Dyn { bounds } => {
                    let iface = bounds.iter().find_map(|b| match b.kind {
                        HirTypeKind::DynInterface { name, .. } => Some(name),
                        _ => None,
                    });
                    match iface {
                        Some(name) => {
                            let unmangled =
                                structs.get(&name).map(|s| s.unmangled_name).unwrap_or(name);
                            SsaType::Pointer(
                                ptr_kind,
                                Box::new(SsaType::Interface(name, unmangled)),
                            )
                        }
                        None => SsaType::Pointer(ptr_kind, Box::new(SsaType::Dyn)),
                    }
                }
                _ => SsaType::Pointer(
                    ptr_kind,
                    Box::new(lower_pointee(inner, enums, structs, in_progress)),
                ),
            }
        }
        HirTypeKind::OwnedPointer { inner, .. } => SsaType::Owned(Box::new(lower_type_hir_inner(
            inner,
            enums,
            structs,
            in_progress,
        ))),
        HirTypeKind::Lambda {
            params,
            return_type,
        } => SsaType::FuncPointer {
            params: params
                .iter()
                .map(|p| lower_type_hir_inner(p, enums, structs, in_progress))
                .collect(),
            return_type: Box::new(lower_type_hir_inner(
                return_type,
                enums,
                structs,
                in_progress,
            )),
        },
        HirTypeKind::Generic(name) => panic!(
            "[lower_type_hir] unsubstituted generic parameter `{}` reached MIR lowering; \
             monomorphization should have resolved every HirType::Generic before this point",
            name
        ),
        HirTypeKind::This => SsaType::Dyn,
        HirTypeKind::Null => SsaType::Void,
        HirTypeKind::Char => SsaType::Char,
        HirTypeKind::Ref {
            inner,
            ref_kind,
            provenance: _,
        } => {
            let ptr_kind = match ref_kind {
                crate::hir::RefKind::Shared => crate::ssa_ir::SsaPointerKind::RefShared,
                crate::hir::RefKind::Unique => crate::ssa_ir::SsaPointerKind::RefMut,
                crate::hir::RefKind::Alias => crate::ssa_ir::SsaPointerKind::RefAlias,
            };
            // Same rationale as SafePointer/UnsafePointer above: `&dyn T`
            // is a vtable-dispatched reference, not a pointer to a struct
            // shaped like the interface's own fields.
            match inner.kind {
                HirTypeKind::DynInterface { name, .. } => {
                    let unmangled = structs.get(&name).map(|s| s.unmangled_name).unwrap_or(name);
                    SsaType::Pointer(ptr_kind, Box::new(SsaType::Interface(name, unmangled)))
                }
                HirTypeKind::Dyn { bounds } => {
                    let iface = bounds.iter().find_map(|b| match b.kind {
                        HirTypeKind::DynInterface { name, .. } => Some(name),
                        _ => None,
                    });
                    match iface {
                        Some(name) => {
                            let unmangled =
                                structs.get(&name).map(|s| s.unmangled_name).unwrap_or(name);
                            SsaType::Pointer(
                                ptr_kind,
                                Box::new(SsaType::Interface(name, unmangled)),
                            )
                        }
                        None => SsaType::Pointer(ptr_kind, Box::new(SsaType::Dyn)),
                    }
                }
                _ => SsaType::Pointer(
                    ptr_kind,
                    Box::new(lower_type_hir_inner(inner, enums, structs, in_progress)),
                ),
            }
        }
        HirTypeKind::Nullable(hir_type) => SsaType::Nullable(Box::new(lower_type_hir_inner(
            hir_type,
            enums,
            structs,
            in_progress,
        ))),
        HirTypeKind::Dyn { bounds } => {
            let iface = bounds.iter().find_map(|b| match b.kind {
                HirTypeKind::DynInterface { name, .. } => Some(name),
                _ => None,
            });
            match iface {
                Some(name) => {
                    let unmangled = structs.get(&name).map(|s| s.unmangled_name).unwrap_or(name);
                    SsaType::Interface(name, unmangled)
                }
                None => SsaType::Dyn,
            }
        }
        HirTypeKind::Unknown => unreachable!(),
        HirTypeKind::Tuple(args) => SsaType::Tuple(
            args.iter()
                .map(|arg| lower_type_hir_inner(arg, enums, structs, in_progress))
                .collect(),
        ),
        HirTypeKind::Array(hir_type, length) => SsaType::Array(
            Box::new(lower_type_hir_inner(hir_type, enums, structs, in_progress)),
            length,
        ),
        HirTypeKind::Slice(hir_type) => SsaType::Slice(Box::new(lower_type_hir_inner(
            hir_type,
            enums,
            structs,
            in_progress,
        ))),
        HirTypeKind::Usize => SsaType::Usize,
        HirTypeKind::Isize => SsaType::Isize,
        HirTypeKind::Never => SsaType::Void,
        HirTypeKind::Range { elem, inclusive: _ } => {
            let elem_ssa = lower_type_hir_inner(elem, enums, structs, in_progress);
            SsaType::Tuple(vec![elem_ssa.clone(), elem_ssa])
        }
    }
}

pub fn lower_operator_bin(operator: &Operator) -> BinOp {
    match operator {
        Operator::Add => BinOp::Add,
        Operator::Subtract => BinOp::Sub,
        Operator::Multiply => BinOp::Mul,
        Operator::Divide => BinOp::Div,
        Operator::Modulo => BinOp::Mod,
        Operator::Equals => BinOp::Eq,
        Operator::NotEquals => BinOp::Ne,
        Operator::LessThan => BinOp::Lt,
        Operator::LessThanOrEqual => BinOp::Le,
        Operator::GreaterThan => BinOp::Gt,
        Operator::GreaterThanOrEqual => BinOp::Ge,
        Operator::LogicalAnd => BinOp::LogicalAnd,
        Operator::LogicalOr => BinOp::LogicalOr,
        Operator::BitAnd => BinOp::BitAnd,
        Operator::BitOr => BinOp::BitOr,
        Operator::BitXor => BinOp::BitXor,
        Operator::ShiftLeft => BinOp::ShiftLeft,
        Operator::ShiftRight => BinOp::ShiftRight,
        _ => todo!("Handle when a non-binary operation is passed here"),
    }
}

fn lower_pointee(
    ty: &HirType,
    enums: &HashMap<StrId, HirEnum<'_, '_>>,
    structs: &HashMap<StrId, HirStruct<'_, '_>>,
    ip: &mut Vec<StrId>,
) -> SsaType {
    match ty.kind {
        HirTypeKind::Struct { name, .. } => {
            let unmangled = structs.get(&name).map(|s| s.unmangled_name).unwrap_or(name);
            SsaType::User(name, unmangled, Vec::new())
        }
        HirTypeKind::Enum { name, .. } => {
            let unmangled = enums.get(&name).map(|e| e.unmangled_name).unwrap_or(name);
            SsaType::Enum {
                name,
                unmangled_name: unmangled,
                variants: Vec::new(),
            }
        }
        _ => lower_type_hir_inner(ty, enums, structs, ip),
    }
}

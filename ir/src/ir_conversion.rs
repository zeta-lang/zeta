use crate::{
    hir::{AssignmentOperator, HirEnum, HirStruct, HirType, Operator, StrId},
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
    match ty {
        HirType::I8 => SsaType::I8,
        HirType::I16 => SsaType::I16,
        HirType::I32 => SsaType::I32,
        HirType::I64 => SsaType::I64,
        HirType::I128 => SsaType::I128,
        HirType::U8 => SsaType::U8,
        HirType::U16 => SsaType::U16,
        HirType::U32 => SsaType::U32,
        HirType::U64 => SsaType::U64,
        HirType::U128 => SsaType::U64,
        HirType::F32 => SsaType::F32,
        HirType::F64 => SsaType::F64,
        HirType::Boolean => SsaType::Bool,
        HirType::String => SsaType::String,

        HirType::Struct {
            name, field_types, ..
        } => {
            if in_progress.contains(name) {
                let unmangled = structs.get(name).map(|s| s.unmangled_name).unwrap_or(*name);
                return SsaType::User(*name, unmangled, Vec::new());
            }

            match structs.get(name).filter(|s| s.generics.is_none()) {
                Some(def) => {
                    in_progress.push(*name);
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
                    SsaType::User(*name, def.unmangled_name, fields)
                }
                // Not registered yet: fall back to what the type carries.
                None => {
                    let mut fields = Vec::with_capacity(field_types.len());
                    for t in field_types.iter() {
                        fields.push(lower_type_hir_inner(t, enums, structs, in_progress));
                    }
                    SsaType::User(*name, *name, fields)
                }
            }
        }

        HirType::DynInterface(name, _args) => {
            let unmangled = structs.get(name).map(|s| s.unmangled_name).unwrap_or(*name);
            SsaType::Interface(*name, unmangled)
        }

        HirType::Enum { name, .. } => {
            let unmangled = enums.get(name).map(|e| e.unmangled_name).unwrap_or(*name);
            if in_progress.contains(name) {
                // Behind a pointer; the layout never looks at the payload.
                return SsaType::Enum {
                    name: *name,
                    unmangled_name: unmangled,
                    variants: Vec::new(),
                };
            }
            let hir_enum = enums.get(name).unwrap_or_else(|| {
                println!("enums: {:?}", enums.keys().collect::<Vec<_>>());
                panic!(
                    "lower_type_hir: enum `{}` not found in registry, it must be \
                             registered (post-monomorphization, under its final concrete name) \
                             before any HirType::Enum referencing it is lowered",
                    name
                )
            });
            in_progress.push(*name);
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
                name: *name,
                unmangled_name: hir_enum.unmangled_name,
                variants,
            }
        }

        HirType::Void => SsaType::Void,
        HirType::SafePointer {
            inner,
            mutability_state,
        }
        | HirType::UnsafePointer {
            inner,
            mutability_state,
        } => {
            // `*dyn T` / `[*]dyn T`: don't drill into the interface's own
            // field layout, that's only meaningful for the implementor.
            // A pointer-to-dyn is vtable-dispatched, same shape regardless
            // of which concrete type is behind it.
            let ptr_kind = if *mutability_state == crate::ast::MutabilityState::Const {
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
            match inner {
                HirType::DynInterface(name, _) => {
                    let unmangled = structs.get(name).map(|s| s.unmangled_name).unwrap_or(*name);
                    SsaType::Pointer(ptr_kind, Box::new(SsaType::Interface(*name, unmangled)))
                }
                HirType::Dyn { bounds } => {
                    let iface = bounds.iter().find_map(|b| match b {
                        HirType::DynInterface(name, _) => Some(*name),
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
        HirType::OwnedPointer { inner, .. } => SsaType::Owned(Box::new(lower_type_hir_inner(
            *inner,
            enums,
            structs,
            in_progress,
        ))),
        HirType::Lambda {
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
        HirType::Generic(name) => panic!(
            "[lower_type_hir] unsubstituted generic parameter `{}` reached MIR lowering; \
             monomorphization should have resolved every HirType::Generic before this point",
            name
        ),
        HirType::This => SsaType::Dyn,
        HirType::Null => SsaType::Void,
        HirType::Char => SsaType::Char,
        HirType::Ref {
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
            match inner {
                HirType::DynInterface(name, _) => {
                    let unmangled = structs.get(name).map(|s| s.unmangled_name).unwrap_or(*name);
                    SsaType::Pointer(ptr_kind, Box::new(SsaType::Interface(*name, unmangled)))
                }
                HirType::Dyn { bounds } => {
                    let iface = bounds.iter().find_map(|b| match b {
                        HirType::DynInterface(name, _) => Some(*name),
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
        HirType::Nullable(hir_type) => SsaType::Nullable(Box::new(lower_type_hir_inner(
            hir_type,
            enums,
            structs,
            in_progress,
        ))),
        HirType::Dyn { bounds } => {
            let iface = bounds.iter().find_map(|b| match b {
                HirType::DynInterface(name, _) => Some(*name),
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
        HirType::Unknown => unreachable!(),
        HirType::Tuple(args) => SsaType::Tuple(
            args.iter()
                .map(|arg| lower_type_hir_inner(arg, enums, structs, in_progress))
                .collect(),
        ),
        HirType::Array(hir_type, length) => SsaType::Array(
            Box::new(lower_type_hir_inner(hir_type, enums, structs, in_progress)),
            *length,
        ),
        HirType::Slice(hir_type) => SsaType::Slice(Box::new(lower_type_hir_inner(
            hir_type,
            enums,
            structs,
            in_progress,
        ))),
        HirType::Usize => SsaType::Usize,
        HirType::Isize => SsaType::Isize,
        HirType::Never => SsaType::Void,
        HirType::Range { elem, inclusive: _ } => {
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
    match ty {
        HirType::Struct { name, .. } => {
            let unmangled = structs.get(name).map(|s| s.unmangled_name).unwrap_or(*name);
            SsaType::User(*name, unmangled, Vec::new())
        }
        HirType::Enum { name, .. } => {
            let unmangled = enums.get(name).map(|e| e.unmangled_name).unwrap_or(*name);
            SsaType::Enum {
                name: *name,
                unmangled_name: unmangled,
                variants: Vec::new(),
            }
        }
        other => lower_type_hir_inner(other, enums, structs, ip),
    }
}

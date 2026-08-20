use crate::{
    hir::{AssignmentOperator, HirEnum, HirType, Operator, StrId},
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

pub fn lower_type_hir(ty: &HirType, enums: &HashMap<StrId, HirEnum<'_, '_>>) -> SsaType {
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
            name,
            field_types: args,
            ..
        } => {
            let lowered_args: Vec<SsaType> =
                args.iter().map(|arg| lower_type_hir(arg, enums)).collect();
            SsaType::User(*name, lowered_args)
        }

        HirType::DynInterface(name, _args) => SsaType::Interface(*name),

        HirType::Enum { name, .. } => {
            let hir_enum = enums.get(name).unwrap_or_else(|| {
                println!("enums: {:#?}", enums.keys().collect::<Vec<_>>());
                panic!(
                    "lower_type_hir: enum `{}` not found in registry, it must be \
                             registered (post-monomorphization, under its final concrete name) \
                             before any HirType::Enum referencing it is lowered",
                    name
                )
            });
            let variants = hir_enum
                .variants
                .iter()
                .map(|v| {
                    v.fields
                        .iter()
                        .map(|f| lower_type_hir(&f.field_type, enums))
                        .collect()
                })
                .collect();
            SsaType::Enum {
                name: *name,
                variants,
            }
        }

        HirType::Void => SsaType::Void,
        HirType::SafePointer { inner, .. } | HirType::UnsafePointer { inner, .. } => {
            // `*dyn T` / `[*]dyn T`: don't drill into the interface's own
            // field layout, that's only meaningful for the implementor.
            // A pointer-to-dyn is vtable-dispatched, same shape regardless
            // of which concrete type is behind it.
            match inner {
                HirType::DynInterface(name, _) => {
                    SsaType::Pointer(Box::new(SsaType::Interface(*name)))
                }
                HirType::Dyn { bounds } => {
                    let iface = bounds.iter().find_map(|b| match b {
                        HirType::DynInterface(name, _) => Some(*name),
                        _ => None,
                    });
                    match iface {
                        Some(name) => SsaType::Pointer(Box::new(SsaType::Interface(name))),
                        None => SsaType::Pointer(Box::new(SsaType::Dyn)),
                    }
                }
                _ => SsaType::Pointer(Box::new(lower_type_hir(inner, enums))),
            }
        }
        HirType::OwnedPointer { inner, .. } => {
            SsaType::Owned(Box::new(lower_type_hir(inner, enums)))
        }
        HirType::Lambda { .. } => {
            unreachable!()
        }
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
            mutability_state: _,
            provenance: _,
        } => {
            // Same rationale as SafePointer/UnsafePointer above: `&dyn T`
            // is a vtable-dispatched reference, not a pointer to a struct
            // shaped like the interface's own fields.
            match inner {
                HirType::DynInterface(name, _) => {
                    SsaType::Pointer(Box::new(SsaType::Interface(*name)))
                }
                HirType::Dyn { bounds } => {
                    let iface = bounds.iter().find_map(|b| match b {
                        HirType::DynInterface(name, _) => Some(*name),
                        _ => None,
                    });
                    match iface {
                        Some(name) => SsaType::Pointer(Box::new(SsaType::Interface(name))),
                        None => SsaType::Pointer(Box::new(SsaType::Dyn)),
                    }
                }
                _ => SsaType::Pointer(Box::new(lower_type_hir(inner, enums))),
            }
        }
        HirType::Nullable(hir_type) => SsaType::Nullable(Box::new(lower_type_hir(hir_type, enums))),
        HirType::Dyn { bounds } => {
            let iface = bounds.iter().find_map(|b| match b {
                HirType::DynInterface(name, _) => Some(*name),
                _ => None,
            });
            match iface {
                Some(name) => SsaType::Interface(name),
                None => SsaType::Dyn,
            }
        }
        HirType::Unknown => unreachable!(),
        HirType::Tuple(args) => {
            SsaType::Tuple(args.iter().map(|arg| lower_type_hir(arg, enums)).collect())
        }
        HirType::Array(hir_type, length) => {
            SsaType::Array(Box::new(lower_type_hir(hir_type, enums)), *length)
        }
        HirType::Slice(hir_type) => SsaType::Slice(Box::new(lower_type_hir(hir_type, enums))),
        HirType::Usize => SsaType::Usize,
        HirType::Isize => SsaType::Isize,
        HirType::Never => SsaType::Void,
        HirType::Range { elem, inclusive: _ } => {
            let elem_ssa = lower_type_hir(elem, enums);
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

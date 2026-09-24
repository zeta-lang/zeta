use ir::hir::{
    HirType, Operator, ProvenanceAnnotation, ProvenancePathSegment, ProvenanceRoot, RefKind, StrId,
};

pub fn operator_symbol(op: &Operator) -> String {
    use Operator::*;
    match op {
        Add => "+",
        Subtract => "-",
        Multiply => "*",
        Divide => "/",
        Modulo => "%",
        Equals => "==",
        NotEquals => "!=",
        LessThan => "<",
        LessThanOrEqual => "<=",
        GreaterThan => ">",
        GreaterThanOrEqual => ">=",
        LogicalAnd => "&&",
        LogicalOr => "||",
        BitAnd => "&",
        BitOr => "|",
        BitXor => "^",
        ShiftLeft => "<<",
        ShiftRight => ">>",
        _ => return format!("{:?}", op), // TODO finish
    }
    .to_string()
}

pub fn str_id_to_string(id: StrId) -> String {
    format!("{}", id)
}

pub fn type_to_string(ty: &HirType) -> String {
    match ty {
        HirType::I8 => "i8".to_string(),
        HirType::I16 => "i16".to_string(),
        HirType::I32 => "i32".to_string(),
        HirType::I64 => "i64".to_string(),
        HirType::U8 => "u8".to_string(),
        HirType::U16 => "u16".to_string(),
        HirType::U32 => "u32".to_string(),
        HirType::U64 => "u64".to_string(),
        HirType::F32 => "f32".to_string(),
        HirType::F64 => "f64".to_string(),
        HirType::I128 => "i128".to_string(),
        HirType::U128 => "u128".to_string(),
        HirType::Boolean => "bool".to_string(),
        HirType::String => "str".to_string(),
        HirType::Void => "void".to_string(),
        HirType::Unknown => "<unknown>".to_string(),
        HirType::Struct {
            name, type_args, ..
        } => {
            if type_args.is_empty() {
                format!("struct {}", str_id_to_string(*name))
            } else {
                let args = type_args
                    .iter()
                    .map(|t| type_to_string(t))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("struct {}<{}>", str_id_to_string(*name), args)
            }
        }
        HirType::DynInterface(name, _) => {
            format!("interface {}", str_id_to_string(*name))
        }
        HirType::Enum {
            name, type_args, ..
        } => {
            if type_args.is_empty() {
                format!("enum {}", str_id_to_string(*name))
            } else {
                let args = type_args
                    .iter()
                    .map(|t| type_to_string(t))
                    .collect::<Vec<_>>()
                    .join(", ");
                format!("enum {}<{}>", str_id_to_string(*name), args)
            }
        }
        HirType::Generic(name) => format!("generic {}", str_id_to_string(*name)),
        HirType::SafePointer {
            inner,
            mutability_state,
        } => format!("*{} {}", mutability_state, inner),
        HirType::UnsafePointer {
            inner,
            mutability_state,
        } => format!("[*]{} {}", mutability_state, inner),
        HirType::Lambda { .. } => "lambda".to_string(),
        HirType::This => "this".to_string(),
        HirType::Null => "null".to_string(),
        HirType::Char => "char".to_string(),
        HirType::Ref {
            inner,
            ref_kind,
            provenance,
        } => {
            let displayed_provenance = if let Some(provenance) = provenance {
                format!("{}", provenance)
            } else {
                String::new()
            };
            if let RefKind::Unique = *ref_kind {
                format!("&{}mut {}", displayed_provenance, inner)
            } else if let RefKind::Alias = *ref_kind {
                format!("&{}alias {}", displayed_provenance, inner)
            } else {
                format!("&{}{}", displayed_provenance, inner)
            }
        }
        HirType::Nullable(hir_type) => format!("?{}", hir_type),
        HirType::Dyn { bounds } => {
            let mut bounds_str = String::new();
            let mut start = true;
            for bound in *bounds {
                bounds_str.push_str(&bound.to_string());
                if start {
                    start = false;
                } else {
                    bounds_str.push_str(" + ");
                }
            }
            format!("dyn {}", bounds_str)
        }
        HirType::Tuple(hir_types) => format!(
            "({})",
            hir_types
                .iter()
                .map(|t| type_to_string(t))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        HirType::Array(inner, len) => format!("[{}]{}", len, type_to_string(inner)),
        HirType::Slice(inner) => format!("[]{}", type_to_string(inner)),
        HirType::OwnedPointer { inner, allocator } => {
            format!(
                "^{} {}",
                allocator
                    .map(|all| provenance_to_string(&all))
                    .unwrap_or(String::from("")),
                type_to_string(inner)
            )
        }
        HirType::Usize => "usize".to_string(),
        HirType::Isize => "isize".to_string(),
        HirType::Never => "never".to_string(),
        HirType::Range { elem, inclusive } => format!(
            "range<{}>{}",
            type_to_string(elem),
            if *inclusive { " (inclusive)" } else { "" }
        ),
    }
}

pub fn provenance_to_string(p: &ProvenanceAnnotation) -> String {
    let root = match p.root {
        ProvenanceRoot::Var(name) => str_id_to_string(name),
        ProvenanceRoot::ThisRoot => "this".to_string(),
        ProvenanceRoot::Global {
            module_idx: _,
            name,
        } => str_id_to_string(name),
        ProvenanceRoot::ImplicitParam(_) => todo!(),
    };
    p.path.iter().fold(root, |acc, seg| match seg {
        ProvenancePathSegment::Field(f) => format!("{}.{}", acc, str_id_to_string(*f)),
        ProvenancePathSegment::Deref => format!("*{}", acc),
    })
}

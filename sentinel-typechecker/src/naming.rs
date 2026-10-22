use ir::hir::{
    HirType, HirTypeKind, JoinStr, Operator, ProvenanceAnnotation, ProvenancePathSegment,
    ProvenanceRoot, RefKind, StrId,
};
use zetaruntime::{intern_fmt, string_pool::StringPool};

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
        Assign => "=",
        AddAssign => "+=",
        SubtractAssign => "-=",
        MultiplyAssign => "*=",
        DivideAssign => "/=",
        ModuloAssign => "%=",
        BitAndAssign => "&=",
        BitOrAssign => "|=",
        BitXorAssign => "^=",
        ShiftLeftAssign => "<<=",
        ShiftRightAssign => ">>=",
        BitNot => "~",
        LogicalNot => "!",
        DerefUnsafe => "*",
        Deref => "*",
        Ref => "&",
        RefMut => "&mut",
    }
    .to_string()
}

pub fn type_to_string(ty: &HirType, pool: &StringPool) -> StrId {
    match ty.kind {
        HirTypeKind::I8 => StrId::from_static("i8"),
        HirTypeKind::I16 => StrId::from_static("i16"),
        HirTypeKind::I32 => StrId::from_static("i32"),
        HirTypeKind::I64 => StrId::from_static("i64"),
        HirTypeKind::U8 => StrId::from_static("u8"),
        HirTypeKind::U16 => StrId::from_static("u16"),
        HirTypeKind::U32 => StrId::from_static("u32"),
        HirTypeKind::U64 => StrId::from_static("u64"),
        HirTypeKind::F32 => StrId::from_static("f32"),
        HirTypeKind::F64 => StrId::from_static("f64"),
        HirTypeKind::I128 => StrId::from_static("i128"),
        HirTypeKind::U128 => StrId::from_static("u128"),
        HirTypeKind::Boolean => StrId::from_static("bool"),
        HirTypeKind::String => StrId::from_static("str"),
        HirTypeKind::Void => StrId::from_static("void"),
        HirTypeKind::Unknown => StrId::from_static("<unknown>"),
        HirTypeKind::Struct {
            unmangled_name: name,
            type_args,
            ..
        } => {
            if type_args.is_empty() {
                StrId(intern_fmt!(pool, "struct {}", name))
            } else {
                let args: String = type_args
                    .iter()
                    .map(|t| type_to_string(t, pool))
                    .collect::<Vec<_>>()
                    .join(", ");
                StrId(intern_fmt!(pool, "struct {}<{}>", name, args))
            }
        }
        HirTypeKind::DynInterface {
            unmangled_name: name,
            ..
        } => StrId(intern_fmt!(pool, "interface {}", name)),
        HirTypeKind::Enum {
            unmangled_name: name,
            type_args,
            ..
        } => {
            if type_args.is_empty() {
                StrId(intern_fmt!(pool, "enum {}", name))
            } else {
                let args: String = type_args
                    .iter()
                    .map(|t| type_to_string(t, pool))
                    .collect::<Vec<_>>()
                    .join(", ");
                StrId(intern_fmt!(pool, "enum {}<{}>", name, args))
            }
        }
        HirTypeKind::Generic(name) => StrId(intern_fmt!(pool, "{}", name)),
        HirTypeKind::SafePointer {
            inner,
            mutability_state,
        } => StrId(intern_fmt!(pool, "*{} {}", mutability_state, inner)),
        HirTypeKind::UnsafePointer {
            inner,
            mutability_state,
        } => StrId(intern_fmt!(pool, "[*]{} {}", mutability_state, inner)),
        HirTypeKind::Lambda { .. } => StrId::from_static("lambda"),
        HirTypeKind::This => StrId::from_static("this"),
        HirTypeKind::Null => StrId::from_static("null"),
        HirTypeKind::Char => StrId::from_static("char"),
        HirTypeKind::Ref {
            inner,
            ref_kind,
            provenance,
        } => {
            let displayed_provenance = if let Some(provenance) = provenance {
                StrId(intern_fmt!(pool, "{}", provenance))
            } else {
                StrId::from_static("")
            };
            if let RefKind::Unique = ref_kind {
                StrId(intern_fmt!(pool, "&{} mut {}", displayed_provenance, inner))
            } else if let RefKind::Alias = ref_kind {
                StrId(intern_fmt!(
                    pool,
                    "&{} alias {}",
                    displayed_provenance,
                    inner
                ))
            } else {
                StrId(intern_fmt!(pool, "&{} {}", displayed_provenance, inner))
            }
        }
        HirTypeKind::Nullable(hir_type) => StrId(intern_fmt!(pool, "?{}", hir_type)),
        HirTypeKind::Dyn { bounds } => {
            let mut bounds_str = String::new();
            let mut start = true;
            for bound in bounds {
                bounds_str.push_str(type_to_string(bound, pool).as_str());
                if start {
                    start = false;
                } else {
                    bounds_str.push_str(" + ");
                }
            }
            StrId(intern_fmt!(pool, "dyn {}", bounds_str))
        }
        HirTypeKind::Tuple(hir_types) => StrId(intern_fmt!(
            pool,
            "({})",
            hir_types
                .iter()
                .map(|t| type_to_string(t, pool))
                .collect::<Vec<_>>()
                .join(", ")
        )),
        HirTypeKind::Array(inner, len) => StrId(intern_fmt!(
            pool,
            "[{}]{}",
            len,
            type_to_string(inner, pool)
        )),
        HirTypeKind::Slice(inner) => StrId(intern_fmt!(pool, "[]{}", type_to_string(inner, pool))),
        HirTypeKind::OwnedPointer { inner, allocator } => StrId(intern_fmt!(
            pool,
            "^{} {}",
            allocator
                .map(|all| provenance_to_string(&all, pool))
                .unwrap_or(StrId::from_static("")),
            type_to_string(inner, pool)
        )),
        HirTypeKind::Usize => StrId::from_static("usize"),
        HirTypeKind::Isize => StrId::from_static("isize"),
        HirTypeKind::Never => StrId::from_static("never"),
        HirTypeKind::Range { elem, inclusive } => StrId(intern_fmt!(
            pool,
            "range<{}>{}",
            type_to_string(elem, pool),
            if inclusive { " (inclusive)" } else { "" }
        )),
    }
}

pub fn provenance_to_string(p: &ProvenanceAnnotation, pool: &StringPool) -> StrId {
    let root = match p.root {
        ProvenanceRoot::Var(name) => name,
        ProvenanceRoot::ThisRoot => StrId::from_static("this"),
        ProvenanceRoot::Global {
            module_idx: _,
            name,
        } => name,
        ProvenanceRoot::ImplicitParam(i) => StrId(intern_fmt!(pool, "{i}")),
        ProvenanceRoot::Static => StrId::from_static("static"),
    };
    p.path.iter().fold(root, |acc, seg| match seg {
        ProvenancePathSegment::Field(f) => StrId(intern_fmt!(pool, "{}.{}", acc, *f)),
        ProvenancePathSegment::Deref => StrId(intern_fmt!(pool, "*{}", acc)),
    })
}

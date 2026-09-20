use crate::{
    ast::{self, MutabilityState},
    hir::{self, HirType, Operator, RefKind, StrId},
};
use smallvec::SmallVec;
use std::sync::Arc;
use zetaruntime::{intern_fmt, string_pool::StringPool};

pub const fn lower_visibility(visibility: &ast::Visibility) -> hir::Visibility {
    match visibility {
        ast::Visibility::Public => hir::Visibility::Public,
        ast::Visibility::Private => hir::Visibility::Private,
        ast::Visibility::Module => hir::Visibility::Module,
        ast::Visibility::Internal => hir::Visibility::Internal,
    }
}

pub const fn lower_cmp_operator(op: ast::Op) -> Operator {
    match op {
        ast::Op::Eq => Operator::Equals,
        ast::Op::Neq => Operator::NotEquals,
        ast::Op::Lt => Operator::LessThan,
        ast::Op::Lte => Operator::LessThanOrEqual,
        ast::Op::Gt => Operator::GreaterThan,
        ast::Op::Gte => Operator::GreaterThanOrEqual,
        _ => unreachable!(),
    }
}

pub fn type_suffix_with_pool(pool: Arc<StringPool>, ty: &HirType) -> StrId {
    StrId(match ty {
        HirType::I32 => pool.intern("i32"),
        HirType::I64 => pool.intern("i64"),
        HirType::U32 => pool.intern("u32"),
        HirType::U64 => pool.intern("u64"),
        HirType::F32 => pool.intern("f32"),
        HirType::F64 => pool.intern("f64"),
        HirType::String => pool.intern("str"),
        HirType::Boolean => pool.intern("boolean"),
        HirType::Struct {
            name, type_args, ..
        } => {
            if type_args.is_empty() {
                // name is already the fully-resolved/mangled identity of this struct
                **name
            } else {
                // only real generic params recurse
                let mut buf: SmallVec<u8, 64> = SmallVec::new();
                buf.extend_from_slice(pool.resolve_bytes(&*name));
                for arg in type_args.iter() {
                    buf.push(b'_');
                    let suf = type_suffix_with_pool(pool.clone(), arg);
                    buf.extend_from_slice(pool.resolve_bytes(&*suf));
                }
                let s = std::str::from_utf8(&buf).expect("valid utf8");
                pool.intern(s)
            }
        }
        HirType::Generic(name) => unreachable!(
            "[type_suffix_with_pool] unresolved generic parameter `{}` reached name mangling",
            name
        ),
        HirType::Void => pool.intern("void"),
        HirType::I8 => pool.intern("i8"),
        HirType::I16 => pool.intern("i16"),
        HirType::U8 => pool.intern("u8"),
        HirType::U16 => pool.intern("u16"),
        HirType::I128 => pool.intern("i128"),
        HirType::U128 => pool.intern("u128"),

        HirType::SafePointer {
            inner,
            mutability_state,
        } => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            let tag = if *mutability_state == MutabilityState::Mut {
                "ptrmut"
            } else {
                "ptrconst"
            };
            intern_fmt!(pool, "{}_{}", tag, pool.resolve_string(&inner_suf))
        }

        HirType::UnsafePointer {
            inner,
            mutability_state,
        } => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            let tag = if *mutability_state == MutabilityState::Mut {
                "uptrmut"
            } else {
                "uptrconst"
            };
            intern_fmt!(pool, "{}_{}", tag, pool.resolve_string(&inner_suf))
        }

        HirType::Ref {
            inner,
            ref_kind,
            provenance: _,
        } => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            let tag = if *ref_kind == RefKind::Unique {
                "refmut"
            } else if *ref_kind == RefKind::Alias {
                "refalias"
            } else {
                "refconst"
            };
            intern_fmt!(pool, "{}_{}", tag, pool.resolve_string(&inner_suf))
        }

        HirType::OwnedPointer { inner, .. } => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            let s = format!("own_{}", pool.resolve_string(&inner_suf));
            pool.intern(&s)
        }

        HirType::Lambda {
            params,
            return_type,
        } => {
            let mut buf: SmallVec<u8, 64> = SmallVec::new();
            buf.extend_from_slice(b"fn");
            for p in params.iter() {
                buf.push(b'_');
                let suf = type_suffix_with_pool(pool.clone(), p);
                buf.extend_from_slice(pool.resolve_bytes(&*suf));
            }
            buf.extend_from_slice(b"_ret_");
            let ret_suf = type_suffix_with_pool(pool.clone(), return_type);
            buf.extend_from_slice(pool.resolve_bytes(&*ret_suf));
            pool.intern_bytes(buf.as_slice())
        }

        HirType::This => unreachable!(
            "[type_suffix_with_pool] `this` type reached name mangling unresolved, \
             should have been substituted with the concrete receiver type first"
        ),

        HirType::Null => pool.intern("null"),

        HirType::Char => pool.intern("char"),

        HirType::Unknown => unreachable!(
            "[type_suffix_with_pool] HirType::Unknown reached name mangling, \
             type checking should have rejected this before monomorphization"
        ),

        HirType::Nullable(inner) => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            intern_fmt!(pool, "opt_{}", pool.resolve_string(&inner_suf))
        }

        HirType::Dyn { bounds } => {
            let mut buf: SmallVec<u8, 64> = SmallVec::new();
            buf.extend_from_slice(b"dyn");
            for b in bounds.iter() {
                buf.push(b'_');
                let suf = type_suffix_with_pool(pool.clone(), b);
                buf.extend_from_slice(pool.resolve_bytes(&*suf));
            }
            pool.intern_bytes(buf.as_slice())
        }

        HirType::Tuple(hir_types) => {
            let mut buf: SmallVec<u8, 64> = SmallVec::new();
            buf.extend_from_slice(b"tuple");
            for t in hir_types.iter() {
                buf.push(b'_');
                let suf = type_suffix_with_pool(pool.clone(), t);
                buf.extend_from_slice(pool.resolve_bytes(&*suf));
            }
            pool.intern_bytes(buf.as_slice())
        }

        HirType::Array(inner, len) => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            intern_fmt!(pool, "arr{}_{}", len, pool.resolve_string(&inner_suf))
        }

        HirType::Slice(inner) => {
            let inner_suf = type_suffix_with_pool(pool.clone(), inner);
            intern_fmt!(pool, "slice_{}", pool.resolve_string(&inner_suf))
        }

        HirType::Usize => pool.intern("usize"),
        HirType::Isize => pool.intern("isize"),

        HirType::DynInterface(name, type_args) => {
            if type_args.is_empty() {
                **name
            } else {
                let mut buf: SmallVec<u8, 64> = SmallVec::new();
                buf.extend_from_slice(pool.resolve_bytes(&*name));
                for arg in type_args.iter() {
                    buf.push(b'_');
                    let suf = type_suffix_with_pool(pool.clone(), arg);
                    buf.extend_from_slice(pool.resolve_bytes(&*suf));
                }
                pool.intern_bytes(buf.as_slice())
            }
        }

        HirType::Enum {
            name, type_args, ..
        } => {
            if type_args.is_empty() {
                // name is already the fully-resolved/mangled identity of this enum,
                // same reasoning as the Struct arm above
                **name
            } else {
                let mut buf: SmallVec<u8, 64> = SmallVec::new();
                buf.extend_from_slice(pool.resolve_bytes(&*name));
                for arg in type_args.iter() {
                    buf.push(b'_');
                    let suf = type_suffix_with_pool(pool.clone(), arg);
                    buf.extend_from_slice(pool.resolve_bytes(&*suf));
                }
                pool.intern_bytes(buf.as_slice())
            }
        }

        HirType::Never => pool.intern("never"),

        HirType::Range { elem, inclusive } => {
            let elem_suf = type_suffix_with_pool(pool.clone(), elem);
            let tag = if *inclusive { "rangeincl" } else { "range" };
            intern_fmt!(pool, "{}_{}", tag, pool.resolve_string(&elem_suf))
        }
    })
}

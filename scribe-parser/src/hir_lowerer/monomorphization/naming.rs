use ir::hir::{HirEnum, HirStruct, HirType, StrId};
use ir::hir_utils::type_suffix_with_pool;
use ir::ir_hasher::{FxHashMap, HashMap};
use smallvec::SmallVec;
use std::sync::Arc;
use zetaruntime::intern_fmt;
use zetaruntime::string_pool::StringPool;

pub fn suffix_for_subs(pool: Arc<StringPool>, subs: &HashMap<StrId, HirType>) -> StrId {
    let mut pieces: Vec<(StrId, StrId)> = subs
        .iter()
        .map(|(k, v)| {
            let suffix = type_suffix_with_pool(pool.clone(), v);
            (*k, suffix)
        })
        .collect();
    // sort deterministically by key bytes so result is stable across hashmap order
    pieces.sort_by_key(|(k, _)| pool.resolve_string(k).to_string());
    let mut buf: SmallVec<u8, 128> = SmallVec::new();
    for (i, (_, v)) in pieces.iter().enumerate() {
        if i > 0 {
            buf.push(b'_');
        }
        buf.extend_from_slice(pool.resolve_bytes(&*v));
    }
    let s = std::str::from_utf8(&buf).expect("Generated string should be valid UTF-8");
    StrId(pool.intern(s))
}

pub fn instantiate_enum_name<'a, 'bump>(
    concrete_args: &[HirType<'a, 'bump>],
    base: &HirEnum<'a, 'bump>,
    context: Arc<StringPool>,
) -> StrId {
    let mut subs = FxHashMap::default();
    if let Some(generics) = &base.generics {
        for (param, arg) in generics.iter().zip(concrete_args) {
            subs.insert(param.name, arg.clone());
        }
    }
    let suffix = suffix_for_subs(context.clone(), &subs);
    StrId(intern_fmt!(context, "{}_{}", base.name, suffix))
}

pub fn instantiate_struct_name(
    concrete_args: &[HirType],
    base: &HirStruct,
    pool: Arc<StringPool>,
) -> StrId {
    let mut buf: SmallVec<u8, 128> = SmallVec::new();
    buf.extend_from_slice(pool.resolve_bytes(&*base.name));
    for arg in concrete_args {
        buf.push(b'_');
        let suf = type_suffix_with_pool(pool.clone(), arg);
        buf.extend_from_slice(pool.resolve_bytes(&*suf));
    }
    let s = std::str::from_utf8(&buf).expect("Generated string should be valid UTF-8");
    let interned = StrId(pool.intern(s));
    interned
}

use arrayvec::ArrayVec;
use codex_dependency_graph::dep_graph::DepGraph;
use ir::hir::JoinStr;
use ir::hir::StrId;
use smallvec::SmallVec;
use std::ptr;
use std::sync::Arc;
use zetaruntime::string_pool::StringPool;

pub fn mangle_method_name(
    dep_graph: &DepGraph,
    module_idx: usize,
    struct_name: StrId,
    method_name: StrId,
    context: Arc<StringPool>,
) -> StrId {
    let mut segments: Vec<StrId> = Vec::with_capacity(4);
    segments.push(struct_name);

    if let Some(pkg) = dep_graph.get_module_package(module_idx) {
        segments.extend(pkg.split("::"));
    }

    build_module_scoped_name(&segments, method_name, None, context)
}

const VTABLE_LEN: usize = 8;

pub fn make_vtable_name(name: StrId, string_pool: Arc<StringPool>) -> StrId {
    let vtable = b"vtable::";
    let nlen = name.len();
    let total_len = VTABLE_LEN + nlen;

    let mut string: SmallVec<u8, 32> = SmallVec::with_capacity(total_len);

    unsafe {
        let dst = string.as_mut_ptr();
        ptr::copy_nonoverlapping(vtable.as_ptr(), dst, VTABLE_LEN);
        ptr::copy_nonoverlapping(name.as_str().as_ptr(), dst.add(VTABLE_LEN), nlen);
        string.set_len(total_len);
    }

    StrId(string_pool.intern_bytes(string.as_slice()))
}

pub fn get_type(name: StrId, string_pool: Arc<StringPool>) -> StrId {
    let mut parts: ArrayVec<&str, 2> = ArrayVec::<&str, 2>::new();
    for part in name.split('_') {
        if parts.try_push(part.as_str()).is_err() {
            panic!("Unexpected type name with too many parts: {}", name);
        }
    }

    if parts.len() == 1 {
        return StrId(string_pool.intern(parts.first().unwrap()));
    }

    if let Some(inner_value) = parts.last().copied() {
        StrId(string_pool.intern(inner_value))
    } else {
        panic!("FieldAccess on non-user pointer type")
    }
}

pub fn build_module_scoped_name(
    path: &[StrId],
    member: StrId,
    extra: Option<StrId>,
    context: Arc<StringPool>,
) -> StrId {
    let mut parts: Vec<StrId> = path.iter().copied().collect();
    parts.push(member);
    if let Some(e) = extra {
        parts.push(e);
    }
    let joined = parts.join("_");
    StrId(context.intern(&joined))
}

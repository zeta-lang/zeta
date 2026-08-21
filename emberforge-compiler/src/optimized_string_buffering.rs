use codex_dependency_graph::DepGraph;
use ir::hir::StrId;
use smallvec::SmallVec;
use std::cell::RefCell;
use std::ptr;
use std::sync::Arc;
use zetaruntime::string_pool::StringPool;

const VTABLE_LEN: usize = 8;

pub fn make_vtable_name(name: StrId, string_pool: Arc<StringPool>) -> StrId {
    let name = string_pool.resolve_string(&*name);

    let vtable = b"vtable::";
    let nlen = name.len();
    let total_len = VTABLE_LEN + nlen;

    let mut string: SmallVec<u8, 32> = SmallVec::with_capacity(total_len);

    unsafe {
        let dst = string.as_mut_ptr();
        ptr::copy_nonoverlapping(vtable.as_ptr(), dst, VTABLE_LEN);
        ptr::copy_nonoverlapping(name.as_ptr(), dst.add(VTABLE_LEN), nlen);
        string.set_len(total_len);
    }

    StrId(string_pool.intern_bytes(string.as_slice()))
}

pub fn build_module_scoped_name(
    path: &[StrId],
    member: StrId,
    extra: Option<StrId>,
    context: Arc<StringPool>,
) -> StrId {
    let mut parts: Vec<String> = path
        .iter()
        .map(|s| context.resolve_string(s).to_string())
        .collect();
    parts.push(context.resolve_string(&member).to_string());
    if let Some(e) = extra {
        parts.push(context.resolve_string(&e).to_string());
    }
    let joined = parts.join("_");
    StrId(context.intern(&joined))
}

pub fn mangle_method_name(
    dep_graph: &RefCell<DepGraph>,
    module_idx: usize,
    struct_name: StrId,
    method_name: StrId,
    context: Arc<StringPool>,
) -> StrId {
    dep_graph
        .borrow()
        .mangle_struct_method(module_idx, struct_name, method_name, &context)
}

pub fn mangle_function_name(
    dep_graph: &DepGraph,
    module_idx: usize,
    struct_name: Option<StrId>,
    func_name: StrId,
    is_extern_c: bool,
    context: Arc<StringPool>,
) -> StrId {
    match struct_name {
        Some(cls) => dep_graph.mangle_struct_method(module_idx, cls, func_name, &context),
        None => dep_graph.mangle_free_function(module_idx, func_name, is_extern_c, &context),
    }
}

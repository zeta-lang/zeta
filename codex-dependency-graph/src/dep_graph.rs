#![allow(unused)]

use ir::hir::JoinStr;
use std::path::PathBuf;

use ir::{
    ast::{Expr, Stmt, Type, TypeKind},
    hir::StrId,
    ir_hasher::{HashMap, HashSet},
    span::SourceSpan,
};
use zetaruntime::string_pool::StringPool;

use crate::{
    nodes::{
        DepNode, Edge, EdgeKind, ItemKey, MethodKey, MethodLookupKey, ModuleIdx, NodeIdx, NodeKind,
        SymbolKey,
    },
    sccs,
};

const PACKAGE_FILE_NAME: &str = "package.zeta";
const MAX_REEXPORT_DEPTH: usize = 16;

fn is_package_file(path: &std::path::Path) -> bool {
    path.file_name().and_then(|n| n.to_str()) == Some(PACKAGE_FILE_NAME)
}

#[derive(Debug)]
pub struct PackageMismatch {
    pub module_idx: usize,
    pub file_path: PathBuf,
    pub declared_package: String,
    pub expected_package: String,
}

impl std::fmt::Display for PackageMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: package mismatch: declared as `package {};`, but its location under \
             the source root implies `package {};`",
            self.file_path.display(),
            self.declared_package,
            self.expected_package,
        )
    }
}

pub struct AstModule<'a, 'bump> {
    pub name: StrId,
    pub path: PathBuf,
    pub stmts: &'bump [Stmt<'a, 'bump>],
}

#[derive(Debug, Clone)]
pub struct UnresolvedImport {
    pub from_module_idx: usize,
    pub path: Vec<StrId>,
}

#[derive(Default, Debug)]
struct PathIndex {
    map: HashMap<Vec<StrId>, usize>,
}

impl PathIndex {
    fn new() -> Self {
        PathIndex {
            map: HashMap::default(),
        }
    }

    fn insert(&mut self, path: Vec<StrId>, module_idx: usize) {
        self.map.insert(path, module_idx);
    }

    fn resolve(&self, path: &[StrId]) -> Option<usize> {
        self.map.get(path).copied()
    }
}

type NodeList<'a> = Vec<DepNode<'a>>;

impl<'a> std::ops::IndexMut<NodeIdx> for Vec<DepNode<'a>> {
    fn index_mut(&mut self, index: NodeIdx) -> &mut Self::Output {
        &mut self[index.0]
    }
}

impl<'a> std::ops::Index<NodeIdx> for Vec<DepNode<'a>> {
    type Output = DepNode<'a>;

    fn index(&self, index: NodeIdx) -> &Self::Output {
        &self[index.0]
    }
}

#[derive(Default, Debug)]
pub struct DepGraph<'a> {
    nodes: NodeList<'a>,

    item_index: HashMap<ItemKey, NodeIdx>,
    symbol_table: HashMap<SymbolKey, ItemKey>,
    method_index: HashMap<MethodKey, NodeIdx>,
    method_symbol_table: HashMap<MethodLookupKey, MethodKey>,
    package_hierarchy: HashMap<ModuleIdx, StrId>,
    package_segments: HashMap<ModuleIdx, Vec<StrId>>,
    module_paths: HashMap<ModuleIdx, PathBuf>,
    package_facades: HashSet<ModuleIdx>,
    facade_exports: HashMap<(ModuleIdx, StrId), Vec<StrId>>,
    current_locals: HashMap<StrId, ItemKey>,
    current_self_type: Option<ItemKey>,

    path_index: PathIndex,

    unresolved_imports: Vec<UnresolvedImport>,
    auto_import_modules: HashSet<usize>,
}

impl<'a> DepGraph<'a> {
    pub fn new() -> Self {
        DepGraph {
            nodes: Vec::new(),
            item_index: HashMap::default(),
            symbol_table: HashMap::default(),
            package_hierarchy: HashMap::default(),
            path_index: PathIndex::new(),
            module_paths: HashMap::default(),
            unresolved_imports: Vec::new(),
            method_index: HashMap::default(),
            method_symbol_table: HashMap::default(),
            current_locals: HashMap::default(),
            current_self_type: None,
            package_segments: HashMap::default(),
            facade_exports: HashMap::default(),
            package_facades: HashSet::default(),
            auto_import_modules: HashSet::default(),
        }
    }

    pub fn register_import(&mut self, current_module_idx: usize, imported_module_idx: usize) {
        todo!()
    }

    pub fn register_package(
        &mut self,
        module_idx: usize,
        path_segments: Vec<ir::hir::StrId>,
        path_strid: ir::hir::StrId,
    ) {
        todo!()
    }

    pub fn register_item_node(
        &mut self,
        module_idx: usize,
        item_idx: usize,
        tag: &'static str,
        node_idx: NodeIdx,
    ) {
        todo!()
    }

    pub fn register_module_structure<'r, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'r, 'bump>,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    pub fn module_symbols(
        &self,
        module_idx: usize,
    ) -> impl Iterator<Item = (ir::hir::StrId, &'static str)> + '_ {
        Vec::new().into_iter()
    }

    pub fn resolve_item_in_module(
        &self,
        module_idx: usize,
        name: ir::hir::StrId,
    ) -> Option<(usize, usize, &'static str)> {
        todo!()
    }

    pub fn remove_module(&mut self, module_idx: usize) {
        todo!()
    }

    /// Remove all
    pub fn remove_file_items(&mut self, module_idx: usize) {
        todo!()
    }

    pub fn extract_edges_for_module<'r, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'r, 'bump>,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    pub fn update_module_items<'r, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'r, 'bump>,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    pub fn module_path(&self, module_idx: usize) -> Option<&std::path::Path> {
        todo!()
    }

    pub fn reverse_deps_transitive_modules(
        &self,
        module_idx: usize,
    ) -> ir::ir_hasher::HashSet<usize> {
        todo!()
    }

    pub fn resolve_module_path(&self, path: &[ir::hir::StrId]) -> Option<usize> {
        todo!()
    }

    pub fn resolve_global_const(
        &self,
        module_idx: usize,
        name: ir::hir::StrId,
    ) -> Option<(usize, usize)> {
        todo!()
    }

    pub fn resolve_method(
        &self,
        target_type: ir::hir::StrId,
        method_name: ir::hir::StrId,
    ) -> Option<(usize, usize, usize)> {
        todo!()
    }

    pub fn resolve_function_in_module(
        &self,
        module_idx: usize,
        name: ir::hir::StrId,
    ) -> Option<(usize, usize)> {
        todo!()
    }

    pub fn find_function_by_name_anywhere(&self, name: ir::hir::StrId) -> Vec<usize> {
        todo!()
    }

    pub fn push_node(&mut self, kind: NodeKind, hint: Option<ir::hir::StrId>) -> NodeIdx {
        todo!()
    }

    pub fn lookup_item_node(
        &self,
        module_idx: usize,
        item_idx: usize,
        tag: &'static str,
    ) -> Option<NodeIdx> {
        todo!()
    }

    pub fn add_edge(&mut self, a: NodeIdx, b: NodeIdx, kind: EdgeKind, span: SourceSpan<'a>) {
        if a.0 >= self.nodes.len() || b.0 >= self.nodes.len() {
            return;
        }
        if !self.nodes[a]
            .deps
            .iter()
            .any(|e| e.to == b && e.kind == kind)
        {
            self.nodes[a].deps.push(Edge { to: b, kind, span });
        }
        if !self.nodes[b].rev_deps.contains(&a) {
            self.nodes[b].rev_deps.push(a);
        }
    }

    pub fn nodes(&self) -> &[DepNode<'a>] {
        &self.nodes
    }

    pub fn package_segments(&self, module_idx: usize) -> Option<&[ir::hir::StrId]> {
        todo!()
    }

    pub fn has_unresolved_imports(&self) -> bool {
        todo!()
    }

    pub fn unresolved_import_messages(&self) -> Vec<String> {
        todo!()
    }

    pub fn check_package_paths(&self, roots: &[std::path::PathBuf]) -> Vec<PackageMismatch> {
        todo!()
    }

    pub fn build_symbol_table(
        &self,
    ) -> ir::ir_hasher::HashMap<(ir::hir::StrId, usize), (usize, usize, &'static str)> {
        todo!()
    }

    pub fn link_stdlib_to_user(&mut self, stdlib_module_idx: usize, user_module_indices: &[usize]) {
        todo!()
    }

    pub fn get_module_imports(&self, module_idx: usize) -> Vec<usize> {
        todo!()
    }

    pub fn get_module_importers(&self, module_idx: usize) -> Vec<usize> {
        todo!()
    }

    pub fn get_module_package(&self, module_idx: usize) -> Option<ir::hir::StrId> {
        todo!()
    }

    pub fn resolve_name(
        &self,
        name: ir::hir::StrId,
        current_module_idx: usize,
        symbol_table: &ir::ir_hasher::HashMap<
            (ir::hir::StrId, usize),
            (usize, usize, &'static str),
        >,
    ) -> Option<NodeIdx> {
        todo!()
    }

    pub fn find_sccs_of_nodes(&self) -> Vec<Vec<NodeIdx>> {
        sccs::find_sccs_of_nodes(&self.nodes)
    }

    pub fn get_module_compilation_order(&self) -> Vec<usize> {
        todo!()
    }

    pub fn get_function_callers(
        &self,
        target_module: usize,
        target_item: usize,
    ) -> Vec<(usize, usize)> {
        todo!()
    }

    pub fn resolve_reexport(&self, module_idx: usize, name: ir::hir::StrId) -> Option<usize> {
        todo!()
    }

    pub fn canonical_member_module(&self, module_idx: usize, name: ir::hir::StrId) -> usize {
        todo!()
    }

    pub fn is_package_facade(&self, module_idx: usize) -> bool {
        todo!()
    }

    pub fn get_function_callees(
        &self,
        caller_module: usize,
        caller_item: usize,
    ) -> Vec<(usize, usize)> {
        todo!()
    }

    pub fn get_function_type_deps(
        &self,
        module_idx: usize,
        item_idx: usize,
    ) -> Vec<(usize, usize)> {
        todo!()
    }

    pub fn can_access(
        &self,
        from_module: usize,
        target_module: usize,
        _target_item_idx: usize,
    ) -> bool {
        todo!()
    }

    pub fn debug_node(&self, node_idx: NodeIdx) -> String {
        todo!()
    }

    pub fn modules_in_same_scc(&self, module_idx: usize) -> Vec<usize> {
        todo!()
    }

    pub fn debug_dump_sccs(&self) {
        todo!()
    }

    pub fn debug_dump_module_order(&self, module_names: &ir::ir_hasher::HashMap<usize, String>) {
        todo!()
    }

    pub fn auto_import_packages(&self) -> Vec<Vec<ir::hir::StrId>> {
        todo!()
    }

    pub fn package_mangle_segments(&self, module_idx: usize) -> Vec<StrId> {
        match self.get_module_package(module_idx) {
            Some(pkg) => pkg.split("::").collect(),
            None => Vec::new(),
        }
    }

    pub fn mangle_free_function(
        &self,
        module_idx: usize,
        name: StrId,
        is_extern_c: bool,
        pool: &StringPool,
    ) -> StrId {
        if is_extern_c {
            return name;
        }
        let segments = self.package_mangle_segments(module_idx);
        Self::scoped_name(&segments, name, pool)
    }

    pub fn mangle_type_name(&self, module_idx: usize, name: StrId, pool: &StringPool) -> StrId {
        let segments = self.package_mangle_segments(module_idx);
        Self::scoped_name(&segments, name, pool)
    }

    pub fn mangle_struct_method(
        &self,
        module_idx: usize,
        struct_name: StrId,
        method_name: StrId,
        pool: &StringPool,
    ) -> StrId {
        let mut segments = Vec::with_capacity(4);
        segments.push(struct_name);
        segments.extend(self.package_mangle_segments(module_idx));
        Self::scoped_name(&segments, method_name, pool)
    }

    fn scoped_name(segments: &[StrId], name: StrId, pool: &StringPool) -> StrId {
        let mut joined: String = segments.join("_");
        if !joined.is_empty() {
            joined.push('_');
        }
        joined.push_str(name.as_str());
        StrId(pool.intern(&joined))
    }

    fn builtin_type_strid(&self, kind: &TypeKind) -> Option<StrId> {
        let s = match kind {
            TypeKind::I8 => "i8",
            TypeKind::I16 => "i16",
            TypeKind::I32 => "i32",
            TypeKind::I64 => "i64",
            TypeKind::U8 => "u8",
            TypeKind::U16 => "u16",
            TypeKind::U32 => "u32",
            TypeKind::U64 => "u64",
            TypeKind::I128 => "i128",
            TypeKind::U128 => "u128",
            TypeKind::F32 => "f32",
            TypeKind::F64 => "f64",
            TypeKind::Usize => "usize",
            TypeKind::Isize => "isize",
            TypeKind::Boolean => "bool",
            TypeKind::String => "str",
            TypeKind::Char => "char",
            _ => return None,
        };
        Some(StrId::from_static(s))
    }

    fn type_name_of<'bump>(&self, ty: &Type<'a, 'bump>) -> Option<StrId> {
        match &ty.kind {
            TypeKind::Struct { name, path, .. } if path.is_empty() => Some(*name),
            other => self.builtin_type_strid(other),
        }
    }
}

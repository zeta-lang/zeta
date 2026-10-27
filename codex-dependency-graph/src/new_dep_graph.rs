use std::path::PathBuf;

use ir::{
    ast::Stmt,
    hir::StrId,
    ir_hasher::{HashMap, HashSet},
};

use crate::{
    nodes::{DepNode, NodeKind},
    sccs,
};

const PACKAGE_FILE_NAME: &str = "package.zeta";
const MAX_REEXPORT_DEPTH: usize = 16;

pub type NodeIdx = usize;
type ItemKey = (usize, usize, &'static str);

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
    /// The module that contains the import statement.
    pub from_module_idx: usize,
    /// The full path segments as written in source, e.g. `["zeta", "io", "File"]`.
    pub path: Vec<StrId>,
}

#[derive(Default, Debug)]
struct PathIndex {
    /// key: Vec<StrId> representing the full qualified path of a module,
    /// value: module_idx
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

#[derive(Default, Debug)]
pub struct DepGraph {
    nodes: Vec<DepNode>,

    /// (module_idx, item_idx, role) -> NodeIdx
    item_index: HashMap<ItemKey, NodeIdx>,

    /// (name StrId, module_idx) -> (module_idx, item_idx, role)
    /// Used for name-based resolution within / across modules.
    symbol_table: HashMap<(StrId, usize), (usize, usize, &'static str)>,

    /// module_idx -> package path StrId (the `package a::b::c` declaration)
    package_hierarchy: HashMap<usize, StrId>,

    /// Fast path-segment lookup built from package declarations.
    path_index: PathIndex,

    module_paths: HashMap<usize, PathBuf>,

    /// Imports that could not be resolved during graph construction.
    pub unresolved_imports: Vec<UnresolvedImport>,

    /// (module_idx, impl_item_idx, method_idx) -> NodeIdx
    method_index: HashMap<(usize, usize, usize), NodeIdx>,

    /// (declaring type name, method name) -> (module_idx, impl_item_idx, method_idx).
    /// Global by design, methods aren't module-scoped for lookup the way
    /// free functions are (a call site knows the receiver's type, not
    /// which module declared the impl).
    method_symbol_table: HashMap<(StrId, StrId), (usize, usize, usize)>,

    current_locals: HashMap<StrId, StrId>,
    current_self_type: Option<StrId>,
    package_segments: HashMap<usize, Vec<StrId>>,
    /// Modules that are `package.zeta` facades.
    package_facades: HashSet<usize>,
    /// (facade module, exported name) -> module path the name is imported from.
    /// Stored as raw segments and resolved lazily, so file load order is irrelevant.
    facade_exports: HashMap<(usize, StrId), Vec<StrId>>,
    auto_import_modules: HashSet<usize>,
}

impl DepGraph {
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

    pub fn package_mangle_segments(&self, module_idx: usize) -> Vec<ir::hir::StrId> {
        todo!()
    }

    pub fn mangle_free_function(
        &self,
        module_idx: usize,
        name: ir::hir::StrId,
        is_extern_c: bool,
        pool: &zetaruntime::string_pool::StringPool,
    ) -> ir::hir::StrId {
        todo!()
    }

    pub fn mangle_type_name(
        &self,
        module_idx: usize,
        name: ir::hir::StrId,
        pool: &zetaruntime::string_pool::StringPool,
    ) -> ir::hir::StrId {
        todo!()
    }

    pub fn mangle_struct_method(
        &self,
        module_idx: usize,
        struct_name: ir::hir::StrId,
        method_name: ir::hir::StrId,
        pool: &zetaruntime::string_pool::StringPool,
    ) -> ir::hir::StrId {
        todo!()
    }

    fn scoped_name(
        segments: &[ir::hir::StrId],
        name: ir::hir::StrId,
        pool: &zetaruntime::string_pool::StringPool,
    ) -> ir::hir::StrId {
        todo!()
    }

    fn builtin_type_strid(&self, kind: &ir::ast::TypeKind) -> Option<ir::hir::StrId> {
        todo!()
    }

    fn type_name_of<'a, 'bump>(&self, ty: &ir::ast::Type<'a, 'bump>) -> Option<ir::hir::StrId> {
        todo!()
    }

    fn expr_type_name<'a, 'bump>(&self, expr: &ir::ast::Expr<'a, 'bump>) -> Option<ir::hir::StrId> {
        todo!()
    }

    fn record_method_call_dep<'a, 'bump>(
        &mut self,
        object: &ir::ast::Expr<'a, 'bump>,
        field: ir::hir::StrId,
        from_node: NodeIdx,
    ) {
        todo!()
    }

    pub fn register_module_structure<'a, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'a, 'bump>,
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

    fn remove_node(&mut self, idx: NodeIdx) {
        todo!()
    }

    pub fn remove_module(&mut self, module_idx: usize) {
        todo!()
    }

    pub fn remove_module_items(&mut self, module_idx: usize) {
        todo!()
    }

    fn create_nodes_for_module<'a, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'a, 'bump>,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn symbol_entries(&self, only_module: Option<usize>) -> Vec<(ItemKey, NodeIdx)> {
        todo!()
    }

    fn populate_symbol_table_for_module(&mut self, module_idx: usize) {
        todo!()
    }

    pub fn extract_edges_for_module<'a, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'a, 'bump>,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    pub fn update_module_items<'a, 'bump>(
        &mut self,
        module_idx: usize,
        module: &AstModule<'a, 'bump>,
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

    pub fn register_item_node(
        &mut self,
        module_idx: usize,
        item_idx: usize,
        tag: &'static str,
        node_idx: NodeIdx,
    ) {
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

    pub fn add_edge(&mut self, a: NodeIdx, b: NodeIdx) {
        todo!()
    }

    pub fn nodes(&self) -> &[DepNode] {
        todo!()
    }

    pub fn build_from_ast<'a, 'bump>(
        &mut self,
        modules: &[AstModule<'a, 'bump>],
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn phase_a_create_nodes<'a, 'bump>(
        &mut self,
        modules: &[AstModule<'a, 'bump>],
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
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

    fn create_node_for_stmt<'a, 'bump>(
        &mut self,
        module_idx: usize,
        item_idx: usize,
        stmt: &ir::ast::Stmt<'a, 'bump>,
    ) {
        todo!()
    }

    fn phase_b_populate_symbol_table(&mut self) {
        todo!()
    }

    fn phase_c_extract_edges<'a, 'bump>(
        &mut self,
        modules: &[AstModule<'a, 'bump>],
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    pub fn build_symbol_table(
        &self,
    ) -> ir::ir_hasher::HashMap<(ir::hir::StrId, usize), (usize, usize, &'static str)> {
        todo!()
    }

    fn extract_edges_for_stmt<'a, 'bump>(
        &mut self,
        module_idx: usize,
        item_idx: usize,
        stmt: &ir::ast::Stmt<'a, 'bump>,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn populate_locals_from_params<'a, 'bump>(
        &mut self,
        params: Option<&'bump [ir::ast::Param<'a, 'bump>]>,
    ) {
        todo!()
    }

    fn walk_func_signature<'a, 'bump>(
        &mut self,
        f: &ir::ast::FuncDecl<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_func_body<'a, 'bump>(
        &mut self,
        f: &ir::ast::FuncDecl<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_struct_decl<'a, 'bump>(
        &mut self,
        s: &ir::ast::StructDecl<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_enum_decl<'a, 'bump>(
        &mut self,
        e: &ir::ast::EnumDecl<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_interface_decl<'a, 'bump>(
        &mut self,
        i: &ir::ast::InterfaceDecl<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_impl_decl<'a, 'bump>(
        &mut self,
        i: &ir::ast::ImplDecl<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        item_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_const_stmt<'a, 'bump>(
        &mut self,
        c: &ir::ast::ConstStmt<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_field<'a, 'bump>(
        &mut self,
        field: &ir::ast::Field<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_block<'a, 'bump>(
        &mut self,
        block: &ir::ast::Block<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_stmt<'a, 'bump>(
        &mut self,
        stmt: &ir::ast::Stmt<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_else_branch<'a, 'bump>(
        &mut self,
        branch: &ir::ast::ElseBranch<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn walk_expr<'a, 'bump>(
        &mut self,
        expr: &ir::ast::Expr<'a, 'bump>,
        from_node: NodeIdx,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn add_module_path_dep(&mut self, from_node: NodeIdx, segments: &[ir::hir::StrId]) {
        todo!()
    }

    fn add_ast_type_dep<'a, 'bump>(
        &mut self,
        from_node: NodeIdx,
        ty: &ir::ast::Type<'a, 'bump>,
        module_idx: usize,
        pool: &zetaruntime::string_pool::StringPool,
    ) {
        todo!()
    }

    fn resolve_name_to_edge(
        &mut self,
        name: ir::hir::StrId,
        from_node: NodeIdx,
        module_idx: usize,
    ) {
        todo!()
    }

    pub fn register_import(&mut self, current_module_idx: usize, imported_module_idx: usize) {
        todo!()
    }

    fn get_or_create_module_node(&mut self, module_idx: usize) -> NodeIdx {
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

    pub fn scc_topo_order(&self) -> Vec<Vec<NodeIdx>> {
        todo!()
    }

    pub fn detect_circular_imports(&self) -> Vec<Vec<usize>> {
        todo!()
    }

    pub fn detect_recursive_cycles(&self) -> Vec<Vec<(usize, usize)>> {
        todo!()
    }

    pub fn get_compilation_order(&self) -> Vec<Vec<(usize, usize)>> {
        todo!()
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

    fn lookup_symbol(
        &self,
        name: ir::hir::StrId,
        module_idx: usize,
    ) -> Option<(usize, usize, &'static str)> {
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

    fn record_auto_import_eligibility<'a, 'bump>(
        &mut self,
        i: &ir::ast::ImplDecl<'a, 'bump>,
        module_idx: usize,
    ) {
        todo!()
    }

    fn target_declared_in_own_package<'a, 'bump>(
        &self,
        target: &ir::ast::Type<'a, 'bump>,
        module_idx: usize,
    ) -> bool {
        todo!()
    }

    pub fn auto_import_packages(&self) -> Vec<Vec<ir::hir::StrId>> {
        todo!()
    }
}

fn strongconnect(
    v: NodeIdx,
    index: &mut Vec<Option<usize>>,
    lowlink: &mut Vec<usize>,
    onstack: &mut Vec<bool>,
    stack: &mut Vec<NodeIdx>,
    counter: &mut usize,
    nodes: &[DepNode],
    sccs: &mut Vec<Vec<NodeIdx>>,
) {
    todo!()
}

fn path_to_strid<'a, 'bump>(
    path: &ir::ast::Path<'a, 'bump>,
    pool: &zetaruntime::string_pool::StringPool,
) -> ir::hir::StrId {
    todo!()
}

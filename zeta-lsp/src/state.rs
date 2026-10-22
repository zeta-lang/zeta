use ir::ast::Visibility;
use ir::hir::StrId;
use ir::ir_hasher::HashMap;
use lsp_types::{CompletionItem, Uri};
use std::path::PathBuf;
use zeta_compiler_api::Compiler;

pub struct Document {
    pub version: i32,
    pub path: PathBuf,
    pub text: String,
}

/// Importable top-level items of one module, independent of who is asking.
pub struct ModuleCandidates {
    pub package: Option<String>,
    pub package_segments: Option<Vec<StrId>>,
    pub items: Vec<(Visibility, CompletionItem)>,
}

#[derive(Default)]
pub struct CompletionCache {
    /// Number of module slots (indices are dense), so completion needn't walk every occurrence.
    pub module_count: Option<usize>,
    pub modules: HashMap<usize, ModuleCandidates>,
}

impl CompletionCache {
    pub fn invalidate_module(&mut self, m: usize) {
        self.modules.remove(&m);
    }
    /// A module may have appeared or disappeared.
    pub fn invalidate_structure(&mut self) {
        self.module_count = None;
    }
    pub fn invalidate_all(&mut self) {
        self.module_count = None;
        self.modules.clear();
    }
}

pub struct ServerState<'a, 'bump> {
    pub compiler: Compiler<'a, 'bump>,
    pub workspace_root: Option<PathBuf>,
    pub documents: HashMap<Uri, Document>,
    pub should_exit: bool,
    pub completion_cache: CompletionCache,
}

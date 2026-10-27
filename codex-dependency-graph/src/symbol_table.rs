use ir::hir::StrId;
use ir::ir_hasher::FxHashBuilder;
use std::collections::HashSet;
use std::path::PathBuf;
use zetaruntime::bump::GrowableBump;

#[derive(Debug)]
pub struct SymbolsSoA {
    pub names: Vec<StrId>,
    pub kinds: Vec<StrId>,
}

impl SymbolsSoA {
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            kinds: Vec::new(),
        }
    }
}

pub struct ModulesSoA {
    pub names: Vec<StrId>,
    pub paths: Vec<PathBuf>,
    pub symbols: Vec<SymbolsSoA>,
    pub deps: Vec<HashSet<usize, FxHashBuilder>>,
}

impl ModulesSoA {
    pub fn new() -> Self {
        Self {
            names: Vec::new(),
            paths: Vec::new(),
            deps: Vec::new(),
            symbols: Vec::new(),
        }
    }

    pub fn push_module(
        &mut self,
        name: StrId,
        path: PathBuf,
        deps: HashSet<usize, FxHashBuilder>,
        symbols: SymbolsSoA,
    ) -> usize {
        let idx = self.names.len();
        self.names.push(name);
        self.paths.push(path);
        self.deps.push(deps);
        self.symbols.push(symbols);
        idx
    }
}

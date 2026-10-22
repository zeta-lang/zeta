use codex_dependency_graph::DepGraph;
use ir::hir::{HirExpr, StrId};
use ir::ir_hasher::HashMap;

pub struct ConstTable<'a, 'bump> {
    values: HashMap<(usize, StrId), HirExpr<'a, 'bump>>,
    by_name: HashMap<StrId, Vec<usize>>,
    aliases: HashMap<usize, HashMap<StrId, usize>>, // module -> `import a.b` aliases
    named: HashMap<usize, HashMap<StrId, usize>>, // module -> `import a.b.NAME` (canonical module)
}

impl<'a, 'bump> ConstTable<'a, 'bump> {
    pub fn new() -> Self {
        Self {
            values: HashMap::default(),
            by_name: HashMap::default(),
            aliases: HashMap::default(),
            named: HashMap::default(),
        }
    }

    pub fn insert(&mut self, module: usize, name: StrId, value: HirExpr<'a, 'bump>) {
        self.values.insert((module, name), value);
        let mods = self.by_name.entry(name).or_default();
        if !mods.contains(&module) {
            mods.push(module);
        }
    }

    pub fn set_imports(
        &mut self,
        module: usize,
        aliases: HashMap<StrId, usize>,
        named: HashMap<StrId, usize>,
    ) {
        self.aliases.insert(module, aliases);
        self.named.insert(module, named);
    }

    fn get(&self, m: usize, name: StrId) -> Option<(usize, &HirExpr<'a, 'bump>)> {
        self.values.get(&(m, name)).map(|v| (m, v))
    }

    pub fn contains_bare(&self, from: usize, name: StrId) -> bool {
        self.resolve_bare(from, name).is_some()
    }

    /// `NAME` as seen from module `from`: own consts, named imports, imported modules, then
    /// (auto-imports) the only module in the program that declares it.
    pub fn resolve_bare(&self, from: usize, name: StrId) -> Option<(usize, &HirExpr<'a, 'bump>)> {
        if let Some(hit) = self.get(from, name) {
            return Some(hit);
        }
        if let Some(&m) = self.named.get(&from).and_then(|n| n.get(&name)) {
            if let Some(hit) = self.get(m, name) {
                return Some(hit);
            }
        }
        if let Some(al) = self.aliases.get(&from) {
            let mut hits = al.values().filter_map(|&m| self.get(m, name));
            if let Some(first) = hits.next() {
                return hits.all(|h| h.0 == first.0).then_some(first);
            }
        }
        match self.by_name.get(&name)?.as_slice() {
            [m] => self.get(*m, name),
            _ => None,
        }
    }

    /// `alias.NAME` / `a::b.NAME` from module `from`.
    pub fn resolve_access(
        &self,
        from: usize,
        path: &[StrId],
        member: StrId,
        dep: &DepGraph,
    ) -> Option<(usize, &HirExpr<'a, 'bump>)> {
        let module = path
            .last()
            .and_then(|l| self.aliases.get(&from)?.get(l).copied())
            .or_else(|| dep.resolve_module_path(path))?;
        self.get(module, member)
            .or_else(|| self.get(dep.canonical_member_module(module, member), member))
    }
}

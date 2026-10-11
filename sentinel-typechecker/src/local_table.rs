use super::bitset::LocalId;
use ir::hir::StrId;
use ir::ir_hasher::FxHashMap;

#[derive(Default, Clone)]
pub struct FunctionLocalTable {
    str_to_id: FxHashMap<StrId, LocalId>,
    id_to_str: Vec<StrId>,
}

impl FunctionLocalTable {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn names(&self) -> impl Iterator<Item = StrId> + '_ {
        self.id_to_str.iter().copied()
    }

    /// Register a local parameter or declared variable.
    pub fn register(&mut self, name: StrId) -> LocalId {
        if let Some(&id) = self.str_to_id.get(&name) {
            return id;
        }
        let id = LocalId(self.id_to_str.len() as u32);
        self.str_to_id.insert(name, id);
        self.id_to_str.push(name);
        id
    }

    #[inline(always)]
    pub fn get(&self, name: StrId) -> Option<LocalId> {
        self.str_to_id.get(&name).copied()
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.id_to_str.len()
    }
}

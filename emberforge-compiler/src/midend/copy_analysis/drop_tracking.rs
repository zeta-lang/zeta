use std::marker::PhantomData;

use ir::{
    hir::{DropKind, HirExpr, HirStmt, StrId},
    ir_hasher::HashMap,
};

#[derive(Clone, Debug)]
pub struct DropLocal<'a, 'bump> {
    pub name: StrId,
    pub kind: DropKind<'a, 'bump>,
}

#[derive(Clone, Debug)]
pub enum ScopeAction<'a, 'bump> {
    DropLocal(DropLocal<'a, 'bump>),
    Defer(&'bump HirStmt<'a, 'bump>),
}

#[derive(Clone, Debug, Default)]
pub struct DropScope<'a, 'bump> {
    pub actions: Vec<ScopeAction<'a, 'bump>>,
}

/// A fact that may hold on some paths but not others.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tri {
    No,
    Yes,
    Maybe,
}

impl Tri {
    pub fn join(self, other: Tri) -> Tri {
        if self == other { self } else { Tri::Maybe }
    }
}

pub type FieldPath = Vec<StrId>;

#[derive(Clone, Debug)]
pub struct DropLocalState {
    pub moved_whole: Tri,
    /// Keyed by full field path from the local: `a.b.c` => [b, c].
    pub moved_fields: HashMap<FieldPath, Tri>,
    pub default_index: Tri,
    pub indices: HashMap<i64, Tri>,
}

impl Default for DropLocalState {
    fn default() -> Self {
        Self {
            moved_whole: Tri::No,
            moved_fields: HashMap::default(),
            default_index: Tri::Yes,
            indices: HashMap::default(),
        }
    }
}

impl DropLocalState {
    fn index(&self, i: i64) -> Tri {
        self.indices.get(&i).copied().unwrap_or(self.default_index)
    }

    /// Status of `path` itself or any ancestor (moving `a.b` moves `a.b.c`).
    fn path_status(&self, path: &[StrId]) -> Tri {
        let mut acc = Tri::No;
        for n in 1..=path.len() {
            match self
                .moved_fields
                .get(&path[..n])
                .copied()
                .unwrap_or(Tri::No)
            {
                Tri::Yes => return Tri::Yes,
                Tri::Maybe => acc = Tri::Maybe,
                Tri::No => {}
            }
        }
        acc
    }

    /// True if some strict descendant of `path` is (maybe) moved.
    fn has_moved_below(&self, path: &[StrId]) -> bool {
        self.moved_fields
            .iter()
            .any(|(k, t)| *t != Tri::No && k.len() > path.len() && k.starts_with(path))
    }

    fn join(&self, o: &Self) -> Self {
        let mut moved_fields = HashMap::default();
        for k in self.moved_fields.keys().chain(o.moved_fields.keys()) {
            // effective status, so an ancestor move on one side is respected
            moved_fields.insert(k.clone(), self.path_status(k).join(o.path_status(k)));
        }
        let mut indices = HashMap::default();
        for k in self.indices.keys().chain(o.indices.keys()) {
            indices.insert(*k, self.index(*k).join(o.index(*k)));
        }
        Self {
            moved_whole: self.moved_whole.join(o.moved_whole),
            moved_fields,
            default_index: self.default_index.join(o.default_index),
            indices,
        }
    }
}

#[derive(Default, Clone, Debug)]
pub struct DropMoveState<'a, 'bump> {
    pub locals: HashMap<StrId, DropLocalState>,
    phantom_data: PhantomData<&'bump &'a ()>,
}

impl<'a, 'bump> DropMoveState<'a, 'bump> {
    pub fn reset_local(&mut self, name: StrId) {
        self.locals.remove(&name);
    }

    /// Moved-out arrays have no live elements; `= uninit` arrays start the same way.
    pub fn mark_whole_moved(&mut self, name: StrId) {
        let s = self.locals.entry(name).or_default();
        s.moved_whole = Tri::Yes;
        s.default_index = Tri::No;
        s.indices.clear();
        s.moved_fields.clear();
    }

    pub fn mark_whole_uninit(&mut self, name: StrId) {
        self.mark_whole_moved(name);
    }

    pub fn mark_whole_initialized(&mut self, name: StrId) {
        if let Some(s) = self.locals.get_mut(&name) {
            s.moved_whole = Tri::No;
            s.default_index = Tri::Yes;
            s.indices.clear();
        }
    }

    pub fn mark_path_moved(&mut self, name: StrId, path: &[StrId]) {
        let s = self.locals.entry(name).or_default();
        s.moved_fields.retain(|k, _| !k.starts_with(path)); // descendants now redundant
        s.moved_fields.insert(path.to_vec(), Tri::Yes);
    }

    pub fn mark_path_initialized(&mut self, name: StrId, path: &[StrId]) {
        if let Some(s) = self.locals.get_mut(&name) {
            s.moved_fields.retain(|k, _| !k.starts_with(path));
        }
    }

    /// Path itself or an ancestor moved. `Maybe` counts as moved (leak, never double-free).
    pub fn is_path_moved(&self, name: StrId, path: &[StrId]) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |l| l.path_status(path) != Tri::No)
    }

    /// Some strict descendant of `path` is moved => `path` needs a partial drop.
    /// `path = []` means "any field of the local".
    pub fn has_moved_below(&self, name: StrId, path: &[StrId]) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |l| l.has_moved_below(path))
    }

    pub fn mark_field_moved(&mut self, n: StrId, f: StrId) {
        self.mark_path_moved(n, &[f])
    }
    pub fn mark_field_initialized(&mut self, n: StrId, f: StrId) {
        self.mark_path_initialized(n, &[f])
    }
    pub fn is_field_moved(&self, n: StrId, f: StrId) -> bool {
        self.is_path_moved(n, &[f])
    }
    pub fn has_any_field_moves(&self, n: StrId) -> bool {
        self.has_moved_below(n, &[])
    }

    pub fn mark_field_uninit(&mut self, name: StrId, field: StrId) {
        self.mark_field_moved(name, field);
    }

    // Whole/field queries: `Maybe` counts as moved. That leaks on one path
    // instead of double-freeing; it becomes exact once these get drop flags.
    pub fn is_whole_moved(&self, name: StrId) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |l| l.moved_whole != Tri::No)
    }

    // Arrays: exact tri-state.
    pub fn mark_index_initialized(&mut self, name: StrId, index: i64) {
        self.locals
            .entry(name)
            .or_default()
            .indices
            .insert(index, Tri::Yes);
    }

    pub fn index_status(&self, name: StrId, index: i64) -> Tri {
        self.locals.get(&name).map_or(Tri::Yes, |l| l.index(index))
    }

    pub fn is_index_uninit(&self, name: StrId, index: i64) -> bool {
        self.index_status(name, index) == Tri::No
    }

    /// Loop headers: the body may initialise anything not already `Yes`.
    pub fn havoc_indices(&mut self, name: StrId) {
        let s = self.locals.entry(name).or_default();
        let h = |t: Tri| if t == Tri::Yes { Tri::Yes } else { Tri::Maybe };
        s.default_index = h(s.default_index);
        for v in s.indices.values_mut() {
            *v = h(*v);
        }
    }

    pub fn join(&self, other: &Self) -> Self {
        let mut locals = HashMap::default();
        for name in self.locals.keys().chain(other.locals.keys()) {
            let a = self.locals.get(name).cloned().unwrap_or_default();
            let b = other.locals.get(name).cloned().unwrap_or_default();
            locals.insert(*name, a.join(&b));
        }
        Self {
            locals,
            phantom_data: PhantomData,
        }
    }

    pub fn join_all(states: impl IntoIterator<Item = Self>) -> Option<Self> {
        let mut it = states.into_iter();
        let first = it.next()?;
        Some(it.fold(first, |acc, s| acc.join(&s)))
    }
}

pub(crate) fn local_is_droppable<'a, 'bump>(
    scope_stack: &[DropScope<'a, 'bump>],
    name: StrId,
) -> Option<DropKind<'a, 'bump>> {
    scope_stack
        .iter()
        .rev()
        .flat_map(|s| s.actions.iter())
        .find_map(|a| match a {
            ScopeAction::DropLocal(l) if l.name == name => Some(l.kind.clone()),
            _ => None,
        })
}

pub fn peel_or_else<'e, 'a, 'b>(e: &'e HirExpr<'a, 'b>) -> &'e HirExpr<'a, 'b> {
    match e {
        HirExpr::OrElse { value, .. } => peel_or_else(value),
        _ => e,
    }
}

/// `a.b.c` => (a, [b, c]); `this.x` => (this, [x]); bare `a` => (a, []).
pub fn move_path_of(expr: &HirExpr) -> Option<(StrId, Vec<StrId>)> {
    let mut path = Vec::new();
    let mut cur = expr;
    loop {
        match cur {
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                path.push(*field);
                cur = &**object;
            }
            HirExpr::Deref { expr, .. } => cur = &**expr,
            HirExpr::Ident(n, _) => {
                path.reverse();
                return Some((*n, path));
            }
            HirExpr::This { .. } => {
                path.reverse();
                return Some((StrId::from_static("this"), path));
            }
            _ => return None,
        }
    }
}

pub fn record_move_if_any<'a, 'bump>(
    scope_stack: &[DropScope<'a, 'bump>],
    drop_state: &mut DropMoveState<'a, 'bump>,
    expr: &HirExpr,
) {
    let Some((root, path)) = move_path_of(peel_or_else(expr)) else {
        return;
    };
    let this = StrId::from_static("this");
    if path.is_empty() {
        if local_is_droppable(scope_stack, root).is_some() {
            drop_state.mark_whole_moved(root);
        }
    } else if root == this || local_is_droppable(scope_stack, root).is_some() {
        drop_state.mark_path_moved(root, &path);
    }
}

use std::marker::PhantomData;

use ir::{
    hir::{DropKind, HirExpr, StrId},
    ir_hasher::HashMap,
};

#[derive(Clone, Debug)]
pub struct DropLocal<'a, 'bump> {
    pub name: StrId,
    pub kind: DropKind<'a, 'bump>,
}

#[derive(Clone, Debug)]
pub struct DropScope<'a, 'bump> {
    pub locals: Vec<DropLocal<'a, 'bump>>,
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

#[derive(Clone, Debug)]
pub struct DropLocalState {
    pub moved_whole: Tri,
    pub moved_fields: HashMap<StrId, Tri>,
    /// Init status of every array index not listed in `indices`.
    /// `Yes` for ordinary arrays, `No` for `= uninit` or moved-out arrays.
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

    fn join(&self, o: &Self) -> Self {
        let mut moved_fields = HashMap::default();
        for k in self.moved_fields.keys().chain(o.moved_fields.keys()) {
            let a = self.moved_fields.get(k).copied().unwrap_or(Tri::No);
            let b = o.moved_fields.get(k).copied().unwrap_or(Tri::No);
            moved_fields.insert(*k, a.join(b));
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
    /// New binding (also handles shadowing): forget everything about this name.
    pub fn reset_local(&mut self, name: StrId) {
        self.locals.remove(&name);
    }

    /// Moved-out arrays have no live elements; `= uninit` arrays start the same way.
    pub fn mark_whole_moved(&mut self, name: StrId) {
        let s = self.locals.entry(name).or_default();
        s.moved_whole = Tri::Yes;
        s.default_index = Tri::No;
        s.indices.clear();
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

    pub fn mark_field_moved(&mut self, name: StrId, field: StrId) {
        self.locals
            .entry(name)
            .or_default()
            .moved_fields
            .insert(field, Tri::Yes);
    }

    pub fn mark_field_uninit(&mut self, name: StrId, field: StrId) {
        self.mark_field_moved(name, field);
    }

    pub fn mark_field_initialized(&mut self, name: StrId, field: StrId) {
        if let Some(s) = self.locals.get_mut(&name) {
            s.moved_fields.remove(&field);
        }
    }

    // Whole/field queries: `Maybe` counts as moved. That leaks on one path
    // instead of double-freeing; it becomes exact once these get drop flags.
    pub fn is_whole_moved(&self, name: StrId) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |l| l.moved_whole != Tri::No)
    }

    pub fn is_field_moved(&self, name: StrId, field: StrId) -> bool {
        self.locals.get(&name).map_or(false, |l| {
            l.moved_fields.get(&field).copied().unwrap_or(Tri::No) != Tri::No
        })
    }

    pub fn has_any_field_moves(&self, name: StrId) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |s| s.moved_fields.values().any(|t| *t != Tri::No))
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
        .flat_map(|s| s.locals.iter())
        .find(|l| l.name == name)
        .map(|l| l.kind.clone())
}

pub fn record_move_if_any<'a, 'bump>(
    scope_stack: &[DropScope<'a, 'bump>],
    drop_state: &mut DropMoveState<'a, 'bump>,
    expr: &HirExpr,
) {
    match expr {
        HirExpr::Ident(name, _) => {
            if local_is_droppable(scope_stack, *name).is_some() {
                drop_state.mark_whole_moved(*name);
            }
        }
        HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
            match &**object {
                HirExpr::Ident(root, _) => {
                    if local_is_droppable(scope_stack, *root).is_some() {
                        drop_state.mark_field_moved(*root, *field);
                    }
                }
                HirExpr::This { .. } => {
                    drop_state.mark_field_moved(StrId::from_static("this"), *field);
                }
                _ => {}
            }
        }
        _ => {}
    }
}

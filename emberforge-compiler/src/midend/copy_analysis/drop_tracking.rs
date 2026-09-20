use std::marker::PhantomData;

use ir::{
    hir::{DropKind, HirExpr, StrId},
    ir_hasher::{HashMap, HashSet},
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

#[derive(Default, Clone, Debug)]
pub struct DropLocalState {
    pub moved_whole: bool,
    pub moved_fields: HashSet<StrId>,
    pub initialized_indices: HashSet<i64>,
}

#[derive(Default, Clone, Debug)]
pub struct DropMoveState<'a, 'bump> {
    pub locals: HashMap<StrId, DropLocalState>,
    phantom_data: PhantomData<&'bump &'a ()>,
}

impl<'a, 'bump> DropMoveState<'a, 'bump> {
    pub fn mark_whole_moved(&mut self, name: StrId) {
        self.locals.entry(name).or_default().moved_whole = true;
    }

    pub fn mark_field_moved(&mut self, name: StrId, field: StrId) {
        self.locals
            .entry(name)
            .or_default()
            .moved_fields
            .insert(field);
    }

    pub fn has_any_field_moves(&self, name: StrId) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |s| !s.moved_fields.is_empty())
    }

    pub fn is_whole_moved(&self, name: StrId) -> bool {
        self.locals.get(&name).map_or(false, |l| l.moved_whole)
    }

    pub fn mark_index_initialized(&mut self, name: StrId, index: i64) {
        self.locals
            .entry(name)
            .or_default()
            .initialized_indices
            .insert(index);
    }

    pub fn is_index_uninit(&self, name: StrId, index: i64) -> bool {
        self.locals.get(&name).map_or(false, |l| {
            l.moved_whole && !l.initialized_indices.contains(&index)
        })
    }

    pub fn is_field_moved(&self, name: StrId, field: StrId) -> bool {
        self.locals
            .get(&name)
            .map_or(false, |l| l.moved_fields.contains(&field))
    }

    pub fn mark_whole_initialized(&mut self, name: StrId) {
        if let Some(s) = self.locals.get_mut(&name) {
            s.moved_whole = false;
        }
    }

    pub fn mark_field_initialized(&mut self, name: StrId, field: StrId) {
        if let Some(s) = self.locals.get_mut(&name) {
            s.moved_fields.remove(&field);
        }
    }

    pub fn mark_whole_uninit(&mut self, name: StrId) {
        self.mark_whole_moved(name);
    }

    pub fn mark_field_uninit(&mut self, name: StrId, field: StrId) {
        self.mark_field_moved(name, field);
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

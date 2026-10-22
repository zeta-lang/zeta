use super::bitset::{LocalBitSet, LocalId};
use ir::hir::StrId;
use smallvec::SmallVec;

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MoveState {
    /// 1 bit per local: variable is moved completely.
    pub whole_moved: LocalBitSet,
    /// 1 bit per local: at least one field of this local is moved.
    pub any_field_moved: LocalBitSet,
    /// Sparse field moves
    pub field_moves: SmallVec<(LocalId, StrId), 4>,
}

impl MoveState {
    pub fn new() -> Self {
        Self::default()
    }

    #[inline(always)]
    pub fn is_whole_moved(&self, id: LocalId) -> bool {
        self.whole_moved.test(id)
    }

    #[inline(always)]
    pub fn blocks_whole_use(&self, id: LocalId) -> bool {
        // A whole use is blocked if the whole value is moved OR any of its fields are moved!
        self.whole_moved.test(id) || self.any_field_moved.test(id)
    }

    #[inline]
    pub fn is_field_moved(&self, id: LocalId, field: StrId) -> bool {
        if self.whole_moved.test(id) {
            return true;
        }
        if !self.any_field_moved.test(id) {
            return false;
        }
        self.field_moves.iter().any(|&(l, f)| l == id && f == field)
    }

    #[inline(always)]
    pub fn mark_whole_moved(&mut self, id: LocalId) {
        self.whole_moved.set(id);
    }

    #[inline]
    pub fn mark_field_moved(&mut self, id: LocalId, field: StrId) {
        self.any_field_moved.set(id);
        if !self.field_moves.iter().any(|&(l, f)| l == id && f == field) {
            self.field_moves.push((id, field));
        }
    }

    #[inline]
    pub fn clear(&mut self, id: LocalId) {
        self.whole_moved.clear(id);
        self.any_field_moved.clear(id);
        self.field_moves.retain(|&(l, _)| l != id);
    }

    #[inline]
    pub fn join_assign(&mut self, other: &MoveState) {
        self.whole_moved.union_with(&other.whole_moved);
        self.any_field_moved.union_with(&other.any_field_moved);
        for &item in &other.field_moves {
            if !self.field_moves.contains(&item) {
                self.field_moves.push(item);
            }
        }
    }

    #[inline]
    pub fn join(a: &MoveState, b: &MoveState) -> MoveState {
        let mut result = a.clone();
        result.join_assign(b);
        result
    }

    #[inline]
    pub fn is_superset_of(&self, other: &MoveState) -> bool {
        if !self.whole_moved.is_superset_of(&other.whole_moved) {
            return false;
        }
        if !self.any_field_moved.is_superset_of(&other.any_field_moved) {
            return false;
        }
        // Check sparse field moves
        other
            .field_moves
            .iter()
            .all(|item| self.field_moves.contains(item))
    }
}

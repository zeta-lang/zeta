use super::bitset::{LocalBitSet, LocalId};
use crate::initialization::{InitNode, InitStatus, join_init_nodes};
use ir::ir_hasher::FxHashMap;

/// Encoding: all-zero reads as `Initialized` (same as the old map's "missing key").
///   Initialized   = (maybe_uninit 0, definitely_uninit 0)
///   Uninitialized = (1, 1)
///   Maybe         = (1, 0)
/// Join is `OR` on maybe_uninit and `AND` on definitely_uninit.
#[derive(Debug, Clone, Default)]
pub struct FastInitState {
    maybe_uninit: LocalBitSet,
    definitely_uninit: LocalBitSet,
    /// Locals that are not a plain `Whole(..)`: partially initialised structs and
    /// arrays with interval ranges. Authoritative over the bits when present.
    structured: FxHashMap<LocalId, InitNode>,
}

impl PartialEq for FastInitState {
    fn eq(&self, other: &Self) -> bool {
        self.maybe_uninit.same_bits(&other.maybe_uninit)
            && self.definitely_uninit.same_bits(&other.definitely_uninit)
            && self.structured == other.structured
    }
}

impl FastInitState {
    pub fn new() -> Self {
        Self::default()
    }

    fn set_whole(&mut self, id: LocalId, s: &InitStatus) {
        match s {
            InitStatus::Initialized => {
                self.maybe_uninit.clear(id);
                self.definitely_uninit.clear(id);
            }
            InitStatus::Uninitialized => {
                self.maybe_uninit.set(id);
                self.definitely_uninit.set(id);
            }
            InitStatus::Maybe => {
                self.maybe_uninit.set(id);
                self.definitely_uninit.clear(id);
            }
        }
    }

    #[inline]
    pub fn whole_status(&self, id: LocalId) -> InitStatus {
        if !self.maybe_uninit.test(id) {
            InitStatus::Initialized
        } else if self.definitely_uninit.test(id) {
            InitStatus::Uninitialized
        } else {
            InitStatus::Maybe
        }
    }

    /// Owned snapshot. For plain locals this is a trivial `Whole(status)` (no allocation).
    #[inline]
    pub fn node(&self, id: LocalId) -> InitNode {
        match self.structured.get(&id) {
            Some(n) => n.clone(),
            None => InitNode::Whole(self.whole_status(id)),
        }
    }

    pub fn set_node(&mut self, id: LocalId, node: InitNode) {
        match node {
            InitNode::Whole(s) => {
                if !self.structured.is_empty() {
                    self.structured.remove(&id);
                }
                self.set_whole(id, &s);
            }
            other => {
                // bits are only a summary for structured locals
                self.set_whole(id, &InitStatus::Maybe);
                self.structured.insert(id, other);
            }
        }
    }

    #[inline]
    pub fn mark_whole_init(&mut self, id: LocalId) {
        self.set_node(id, InitNode::Whole(InitStatus::Initialized));
    }

    #[inline]
    pub fn mark_whole_uninit(&mut self, id: LocalId) {
        self.set_node(id, InitNode::Whole(InitStatus::Uninitialized));
    }

    #[inline]
    pub fn is_whole_init(&self, id: LocalId) -> bool {
        !self.structured.contains_key(&id) && !self.maybe_uninit.test(id)
    }

    pub fn join(a: &FastInitState, b: &FastInitState) -> FastInitState {
        let mut maybe_uninit = a.maybe_uninit.clone();
        maybe_uninit.union_with(&b.maybe_uninit);
        let mut definitely_uninit = a.definitely_uninit.clone();
        definitely_uninit.intersect_with(&b.definitely_uninit);

        let mut out = FastInitState {
            maybe_uninit,
            definitely_uninit,
            structured: FxHashMap::default(),
        };
        if a.structured.is_empty() && b.structured.is_empty() {
            return out; // the overwhelmingly common case: two bitwise ops
        }
        for &id in a.structured.keys() {
            out.set_node(id, join_init_nodes(a.node(id), b.node(id)));
        }
        for &id in b.structured.keys() {
            if !a.structured.contains_key(&id) {
                out.set_node(id, join_init_nodes(a.node(id), b.node(id)));
            }
        }
        out
    }
}

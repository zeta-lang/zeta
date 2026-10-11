use std::collections::VecDeque;

use crate::{
    bitset::LocalId, initialization::InitNode, initialization_bitset::FastInitState,
    move_state::MoveState,
};
use ir::{
    hir::{HirType, StrId},
    ir_hasher::FxHashMap,
    nll_cfg::PointId,
};

#[derive(Clone)]
pub struct FlowState {
    pub mv: MoveState,
    pub init: FastInitState,
}

impl FlowState {
    pub fn join(a: &Self, b: &Self) -> Self {
        Self {
            mv: MoveState::join(&a.mv, &b.mv),
            init: FastInitState::join(&a.init, &b.init),
        }
    }

    /// Mutual superset, not derived `==`: bitsets can differ in trailing zero words.
    fn same(&self, o: &Self) -> bool {
        self.init == o.init && self.mv.is_superset_of(&o.mv) && o.mv.is_superset_of(&self.mv)
    }
}

/// State-independent operations, recorded by the mutators during the recording walk.
#[derive(Clone)]
pub enum Effect<'a, 'bump> {
    Move {
        local: LocalId,
        field: Option<StrId>,
    },
    ClearMove(LocalId),
    InitWhole(LocalId),
    UninitWhole(LocalId),
    InitField(LocalId, Vec<StrId>),
    UninitField {
        local: LocalId,
        path: Vec<StrId>,
        root_ty: Option<HirType<'a, 'bump>>,
    },
    ArrayRange {
        local: LocalId,
        path: Vec<StrId>,
        start: i64,
        end: i64,
        len: Option<usize>,
    },
    ArrayHole {
        local: LocalId,
        path: Vec<StrId>,
        idx: i64,
    },
    /// Set the node at `path` to a constant (`Whole(Initialized)` / `Whole(Maybe)`).
    SetPath(LocalId, Vec<StrId>, InitNode),
}

pub struct Solved {
    entry_in: Vec<Option<FlowState>>,
    pub(crate) index: FxHashMap<PointId, usize>,
}

impl Solved {
    pub fn entry_in(&self, p: PointId) -> Option<&FlowState> {
        let i = *self.index.get(&p)?;
        self.entry_in.get(i)?.as_ref()
    }
}

pub fn solve(
    n: usize,
    preds: &[Vec<usize>],
    succs: &[Vec<usize>],
    entry: usize,
    entry_state: FlowState,
    transfer: &dyn Fn(usize, &mut FlowState),
) -> Solved {
    let mut inn: Vec<Option<FlowState>> = vec![None; n];
    let mut out: Vec<Option<FlowState>> = vec![None; n];
    let mut queued = vec![true; n];
    let mut wl: VecDeque<usize> = (0..n).collect();

    while let Some(p) = wl.pop_front() {
        queued[p] = false;

        // None = bottom. Never start from FlowState::default(): see note 3 above.
        let mut acc = (p == entry).then(|| entry_state.clone());
        for &q in &preds[p] {
            if let Some(o) = &out[q] {
                acc = Some(match acc {
                    None => o.clone(),
                    Some(a) => FlowState::join(&a, o),
                });
            }
        }
        let Some(in_state) = acc else { continue };

        let mut o = in_state.clone();
        transfer(p, &mut o);
        inn[p] = Some(in_state);

        if out[p].as_ref().map_or(true, |old| !old.same(&o)) {
            out[p] = Some(o);
            for &s in &succs[p] {
                if !queued[s] {
                    queued[s] = true;
                    wl.push_back(s);
                }
            }
        }
    }
    Solved {
        entry_in: inn,
        index: FxHashMap::default(),
    }
}

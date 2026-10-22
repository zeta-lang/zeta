use std::collections::HashMap;

use crate::borrow_checker::{
    AliasReasoner, Bound, IndexContainer, Interval, MemoryRelation, Place, PlaceId, Projection,
};

#[inline(always)]
fn get_place(places: &[Place], id: PlaceId) -> Option<&Place> {
    places.get(id.0 as usize)
}

impl AliasReasoner {
    pub fn new() -> Self {
        Self {
            disjoint_indexable: HashMap::default(),
            equal: HashMap::default(),
            not_equal: HashMap::default(),
            less_than: HashMap::default(),
            ranges: HashMap::default(),
            disjoint_intervals: Vec::default(),
            place_not_equal: HashMap::default(),
        }
    }

    pub(super) fn reset(&mut self) {
        self.disjoint_indexable.clear();
        self.equal.clear();
        self.not_equal.clear();
        self.less_than.clear();
        self.ranges.clear();
        self.disjoint_intervals.clear();
        self.place_not_equal.clear();
    }

    /// True if some ancestor-or-self of `lhs` is recorded as not-equal to some
    /// ancestor-or-self of `rhs`.
    #[inline]
    fn check_not_equal_chains(&self, places: &[Place], lhs: PlaceId, rhs: PlaceId) -> bool {
        let mut l = Some(lhs);
        while let Some(lid) = l {
            if let Some(set) = self.place_not_equal.get(&lid) {
                if !set.is_empty() {
                    let mut r = Some(rhs);
                    while let Some(rid) = r {
                        if set.contains(&rid) {
                            return true;
                        }
                        r = get_place(places, rid).and_then(|p| p.parent);
                    }
                }
            }
            l = get_place(places, lid).and_then(|p| p.parent);
        }
        false
    }

    #[inline(always)]
    pub fn relation(&self, places: &[Place], lhs: PlaceId, rhs: PlaceId) -> MemoryRelation {
        if lhs == rhs {
            return MemoryRelation::Overlap;
        }

        let l_place = &places[lhs.0 as usize];
        let r_place = &places[rhs.0 as usize];

        if l_place.root != r_place.root {
            return MemoryRelation::Disjoint;
        }

        if !self.place_not_equal.is_empty() {
            if self.check_not_equal_chains(places, lhs, rhs) {
                return MemoryRelation::Disjoint;
            }
        }

        match self.find_divergence_fast(places, lhs, rhs) {
            Divergence::AncestorDescendant => MemoryRelation::Overlap,
            Divergence::Diverged(ldiv, rdiv) => {
                let ld = &places[ldiv.0 as usize];
                let rd = &places[rdiv.0 as usize];
                match (&ld.projection, &rd.projection) {
                    (Some(Projection::Field(a)), Some(Projection::Field(b))) => {
                        if a == b {
                            MemoryRelation::Overlap
                        } else {
                            MemoryRelation::Disjoint
                        }
                    }
                    (
                        Some(Projection::Index {
                            index: li,
                            container,
                        }),
                        Some(Projection::Index { index: ri, .. }),
                    ) => self.reason_about_indices(li, ri, *container),
                    _ => MemoryRelation::Unknown,
                }
            }
            Divergence::Disjoint => MemoryRelation::Disjoint,
        }
    }

    #[inline]
    fn find_divergence_fast(&self, places: &[Place], mut a: PlaceId, mut b: PlaceId) -> Divergence {
        let mut depth_a = places[a.0 as usize].depth;
        let mut depth_b = places[b.0 as usize].depth;

        // Equalize depths
        while depth_a > depth_b {
            let parent = match places[a.0 as usize].parent {
                Some(p) => p,
                None => return Divergence::Disjoint,
            };
            if parent == b {
                return Divergence::AncestorDescendant;
            }
            a = parent;
            depth_a -= 1;
        }

        while depth_b > depth_a {
            let parent = match places[b.0 as usize].parent {
                Some(p) => p,
                None => return Divergence::Disjoint,
            };
            if parent == a {
                return Divergence::AncestorDescendant;
            }
            b = parent;
            depth_b -= 1;
        }

        // Walk up together until divergence point right under common ancestor
        while a != b {
            let pa = match places[a.0 as usize].parent {
                Some(p) => p,
                None => return Divergence::Disjoint,
            };
            let pb = match places[b.0 as usize].parent {
                Some(p) => p,
                None => return Divergence::Disjoint,
            };

            if pa == pb {
                return Divergence::Diverged(a, b);
            }
            a = pa;
            b = pb;
        }

        Divergence::AncestorDescendant
    }

    pub fn retract_place_not_equal(&mut self, lhs: &PlaceId, rhs: &PlaceId) {
        if let Some(set) = self.place_not_equal.get_mut(lhs) {
            set.remove(rhs);
        }
        if let Some(set) = self.place_not_equal.get_mut(rhs) {
            set.remove(lhs);
        }
    }

    fn resolve_const(&self, bound: &Bound) -> Option<i64> {
        if let Bound::Const(v) = bound {
            return Some(*v);
        }
        self.equal.get(bound).and_then(|set| {
            set.iter().find_map(|b| match b {
                Bound::Const(v) => Some(*v),
                _ => None,
            })
        })
    }

    fn reason_about_indices(
        &self,
        lhs: &Interval,
        rhs: &Interval,
        container: IndexContainer,
    ) -> MemoryRelation {
        let trusted = match container {
            IndexContainer::Primitive => true,
            IndexContainer::UserDefined(ty) => {
                self.disjoint_indexable.get(&ty).copied().unwrap_or(false)
            }
        };

        if !trusted {
            return MemoryRelation::Unknown;
        }

        if lhs == rhs {
            return MemoryRelation::Overlap;
        }

        let (llo, lhi) = Self::half_open(lhs);
        let (rlo, rhi) = Self::half_open(rhs);

        // Disjoint iff one ends at or before the other starts.
        if self.provably_le(&lhi, &rlo, 0) || self.provably_le(&rhi, &llo, 0) {
            return MemoryRelation::Disjoint;
        }
        // Definitely overlapping iff each starts strictly before the other ends.
        if self.provably_le(&llo, &rhi, 1) && self.provably_le(&rlo, &lhi, 1) {
            return MemoryRelation::Overlap;
        }

        if lhs.lower == rhs.lower {
            return MemoryRelation::Overlap;
        }
        if self
            .equal
            .get(&lhs.lower)
            .is_some_and(|s| s.contains(&rhs.lower))
        {
            return MemoryRelation::Overlap;
        }

        // "different lower bound => disjoint" is only valid for single-element indices.
        let both_points = lhs.lower == lhs.upper && rhs.lower == rhs.upper;
        if both_points {
            if self
                .not_equal
                .get(&lhs.lower)
                .is_some_and(|s| s.contains(&rhs.lower))
            {
                return MemoryRelation::Disjoint;
            }
            if self
                .less_than
                .get(&lhs.lower)
                .is_some_and(|s| s.contains(&rhs.lower))
                || self
                    .less_than
                    .get(&rhs.lower)
                    .is_some_and(|s| s.contains(&lhs.lower))
            {
                return MemoryRelation::Disjoint;
            }
        }

        for (left, right) in &self.disjoint_intervals {
            if self.interval_matches(lhs, rhs, left, right) {
                return MemoryRelation::Disjoint;
            }
        }
        MemoryRelation::Unknown
    }

    fn half_open(iv: &Interval) -> (Bound, Bound) {
        if iv.lower == iv.upper {
            (
                iv.lower.clone(),
                Bound::Offset {
                    base: Box::new(iv.lower.clone()),
                    offset: 1,
                },
            )
        } else {
            (iv.lower.clone(), iv.upper.clone())
        }
    }

    /// Can we prove `a + k <= b`?
    fn provably_le(&self, a: &Bound, b: &Bound, k: i64) -> bool {
        let (ab, ao) = Self::decompose_offset(a);
        let (bb, bo) = Self::decompose_offset(b);
        if let (Some(x), Some(y)) = (self.resolve_const(ab), self.resolve_const(bb)) {
            return x.saturating_add(ao).saturating_add(k) <= y.saturating_add(bo);
        }
        if ab == bb || self.compare_linear(ab, bb) == Some(true) {
            return ao + k <= bo;
        }
        if self.less_than.get(ab).is_some_and(|s| s.contains(bb)) {
            return ao + k <= bo + 1; // a < b means a + 1 <= b
        }
        false
    }

    pub fn retract_equal(&mut self, lhs: &Bound, rhs: &Bound) {
        if let Some(set) = self.equal.get_mut(lhs) {
            set.remove(rhs);
        }
        if let Some(set) = self.equal.get_mut(rhs) {
            set.remove(lhs);
        }
    }

    pub fn retract_not_equal(&mut self, lhs: &Bound, rhs: &Bound) {
        if let Some(set) = self.not_equal.get_mut(lhs) {
            set.remove(rhs);
        }
        if let Some(set) = self.not_equal.get_mut(rhs) {
            set.remove(lhs);
        }
    }

    fn compare_linear(&self, lhs: &Bound, rhs: &Bound) -> Option<bool> {
        use Bound::*;

        if let (Some(a), Some(b)) = (self.resolve_const(lhs), self.resolve_const(rhs)) {
            return Some(a == b);
        }

        match (lhs, rhs) {
            (Const(a), Const(b)) => Some(a == b),
            (Symbol(a), Symbol(b)) if a == b => Some(true),
            (SelfField(a), SelfField(b)) => Some(a == b),

            (
                Scale {
                    base: lb,
                    factor: lf,
                },
                Scale {
                    base: rb,
                    factor: rf,
                },
            ) => {
                if lf != rf {
                    return None;
                }
                if *lf == 0 {
                    return Some(true);
                }
                self.compare_linear(lb, rb)
            }

            (FreshPerCall, FreshPerCall) => Some(false),

            // Different Opaque ids are unknown whether equal
            (Opaque(_), _) | (_, Opaque(_)) => None,

            (Sum(a1, b1), Sum(a2, b2)) => {
                match (self.compare_linear(a1, a2), self.compare_linear(b1, b2)) {
                    (Some(true), Some(true)) => Some(true),
                    _ => None,
                }
            }

            (Offset { .. }, _) | (_, Offset { .. }) => {
                let (lhs_base, lhs_off) = Self::decompose_offset(lhs);
                let (rhs_base, rhs_off) = Self::decompose_offset(rhs);
                match self.compare_linear(lhs_base, rhs_base) {
                    Some(true) => Some(lhs_off == rhs_off),
                    Some(false) | None => None,
                }
            }

            _ => self.constraint_verdict(lhs, rhs),
        }
    }

    fn decompose_offset(bound: &Bound) -> (&Bound, i64) {
        match bound {
            Bound::Offset { base, offset } => (base.as_ref(), *offset),
            other => (other, 0),
        }
    }

    fn interval_matches(
        &self,
        left: &Interval,
        right: &Interval,
        expected_left: &Interval,
        expected_right: &Interval,
    ) -> bool {
        (left == expected_left && right == expected_right)
            || (left == expected_right && right == expected_left)
    }

    fn constraint_verdict(&self, lhs: &Bound, rhs: &Bound) -> Option<bool> {
        if lhs == rhs {
            return Some(true);
        }

        if self.equal.get(lhs).is_some_and(|set| set.contains(rhs)) {
            return Some(true);
        }

        if self.not_equal.get(lhs).is_some_and(|set| set.contains(rhs)) {
            return Some(false);
        }

        None
    }
}

pub enum Divergence {
    AncestorDescendant,
    Diverged(PlaceId, PlaceId),
    Disjoint,
}

use ir::{
    hir::{HirExpr, Operator, StrId},
    ir_hasher::{FxHashMap, HashSet},
};

use crate::TypeChecker;

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn mark_non_null(&mut self, root: StrId, path: &[StrId]) {
        self.non_null_state
            .entry(root)
            .or_default()
            .insert(path.to_vec());
    }

    /// Clears exact-path and every path *below* it (assigning `this.tail = x`
    /// invalidates anything previously proven about `this.tail.next`, etc.)
    pub fn clear_non_null(&mut self, root: StrId, path: &[StrId]) {
        if let Some(set) = self.non_null_state.get_mut(&root) {
            set.retain(|p| !(p.len() >= path.len() && p[..path.len()] == *path));
        }
    }

    pub fn is_non_null(&self, root: StrId, path: &[StrId]) -> bool {
        self.non_null_state
            .get(&root)
            .is_some_and(|set| set.contains(path))
    }

    pub fn join_non_null_states(
        a: &FxHashMap<StrId, HashSet<Vec<StrId>>>,
        b: &FxHashMap<StrId, HashSet<Vec<StrId>>>,
    ) -> FxHashMap<StrId, HashSet<Vec<StrId>>> {
        let mut result = FxHashMap::default();
        for (root, a_paths) in a {
            if let Some(b_paths) = b.get(root) {
                let intersected: HashSet<Vec<StrId>> =
                    a_paths.intersection(b_paths).cloned().collect();
                if !intersected.is_empty() {
                    result.insert(*root, intersected);
                }
            }
        }
        result
    }

    pub fn condition_to_non_null_fact(
        &self,
        cond: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<StrId>, bool)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };
        match op {
            // `x != null` -> non-null holds in the *true* branch.
            Operator::NotEquals
                if matches!(right, HirExpr::Null(_)) || matches!(left, HirExpr::Null(_)) =>
            {
                let non_null_side = if matches!(right, HirExpr::Null(_)) {
                    left
                } else {
                    right
                };
                let (root, path) = self.static_field_path(non_null_side)?;
                Some((root, path, true))
            }
            // `x == null` -> non-null holds in the *false* (else) branch.
            Operator::Equals
                if matches!(right, HirExpr::Null(_)) || matches!(left, HirExpr::Null(_)) =>
            {
                let non_null_side = if matches!(right, HirExpr::Null(_)) {
                    left
                } else {
                    right
                };
                let (root, path) = self.static_field_path(non_null_side)?;
                Some((root, path, false))
            }
            // `x == 5` (nullable-equality) -> non-null holds in the *true* branch;
            // `x != 5` establishes nothing (`x == null` also satisfies `!= 5`).
            Operator::Equals => {
                let non_null_side = match (left, right) {
                    (_e, HirExpr::Null(_)) | (HirExpr::Null(_), _e) => return None, // handled above
                    (e, _other) if self.static_field_path(e).is_some() => e,
                    _ => return None,
                };
                let (root, path) = self.static_field_path(non_null_side)?;
                Some((root, path, true))
            }
            _ => None,
        }
    }
}

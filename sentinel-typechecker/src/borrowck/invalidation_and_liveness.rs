use std::collections::VecDeque;

use ir::{
    borrow_checker::{PlaceId, ProvenanceId},
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, HirFunc, HirParam, HirStmt, HirType, StrId},
    ir_hasher::{HashMap, HashSet},
    nll_cfg::PointId,
};
use smallvec::SmallVec;

use super::hir_visit::{HirVisitor, walk_stmt};
use crate::{TypeChecker, str_id_to_string, type_checker::InvalidationCause};

/// Short-circuiting visitor: "does this expr/stmt mention `target`?"
struct ReferencesLocal {
    target: StrId,
    found: bool,
}

impl<'a, 'bump> HirVisitor<'a, 'bump> for ReferencesLocal {
    fn visit_ident(&mut self, name: StrId) {
        if name == self.target {
            self.found = true;
        }
    }

    fn should_continue(&self) -> bool {
        !self.found
    }
}

/// Fills `point_locals_used`: every identifier is attributed to the CFG point of the
/// innermost enclosing statement that has one (`stmt_points`).
struct LocalsCollector<'t, 'a, 'bump> {
    tc: &'t mut TypeChecker<'a, 'bump>,
}

impl<'t, 'a, 'bump> HirVisitor<'a, 'bump> for LocalsCollector<'t, 'a, 'bump> {
    fn visit_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        if let Some(&point) = self.tc.stmt_points.get(&TypeChecker::stmt_key(stmt)) {
            self.tc.current_point = point;
        }
        walk_stmt(self, stmt);
    }

    fn visit_ident(&mut self, name: StrId) {
        let point = self.tc.current_point;
        self.tc
            .point_locals_used
            .entry(point)
            .or_default()
            .insert(name);
    }
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// Marks every live loan on `place` or anything beneath it as pointing at freed memory.
    /// (Recorded per provenance; the error appears when such a reference is next *used*.)
    pub fn invalidate_loans_under(&mut self, place: PlaceId, cause: InvalidationCause) {
        for loan in self.borrow_checker.active_loans.iter().flatten() {
            if self.place_is_within(loan.place, place) {
                self.invalidated_provenances
                    .insert(loan.provenance_id, cause);
            }
        }
    }

    /// Searches `prov` and all its *parent* provenances (references derived from other references)
    /// for an invalidation. A reference is dead if anything it was derived from is dead.
    pub fn invalidating_call_for(&self, prov: ProvenanceId) -> Option<InvalidationCause> {
        if let Some(&c) = self.invalidated_provenances.get(&prov) {
            return Some(c);
        }
        let pr = self.borrow_checker.provenances.get(&prov)?;
        if pr.parents.is_empty() {
            return None;
        }

        let mut stack: SmallVec<ProvenanceId, 8> = SmallVec::from_slice_copy(&pr.parents);
        let mut seen: HashSet<ProvenanceId> = HashSet::default();
        seen.insert(prov);
        while let Some(p) = stack.pop() {
            if !seen.insert(p) {
                continue;
            }
            if let Some(&c) = self.invalidated_provenances.get(&p) {
                return Some(c);
            }
            if let Some(pr) = self.borrow_checker.provenances.get(&p) {
                stack.extend_from_slice(&pr.parents);
            }
        }
        None
    }

    /// Reports an error if the reference being used (identified by `prov`) was invalidated.
    /// The message depends on why: a call that may reallocate, or the owner being dropped.
    pub fn check_provenance_still_valid(&mut self, prov: Option<ProvenanceId>) {
        let Some(prov) = prov else { return };
        let Some(cause) = self.invalidating_call_for(prov) else {
            return;
        };
        let msg = match cause {
            InvalidationCause::Reallocated(c) => format!(
                "the reference's lifetime has ended because it was invalidated \
                 (`{}` may reallocate the memory it points into)",
                str_id_to_string(c)
            ),
            InvalidationCause::Dropped(t) => format!(
                "the reference's lifetime has ended: the `{}` it was borrowed from \
                 has been dropped (for a lock guard, the lock is released)",
                str_id_to_string(t)
            ),
        };
        self.record(TypeErrorKind::Generic(msg));
    }

    /// Type of `path` (a chain of struct fields) starting at struct `owner`.
    /// `None` if a segment is missing or an intermediate field isn't a struct.
    fn field_type_at(&self, owner: StrId, path: &[StrId]) -> Option<HirType<'a, 'bump>> {
        let mut cur = owner;
        let mut ty = None;
        for (i, seg) in path.iter().enumerate() {
            let def = self.context.get_struct(&str_id_to_string(cur))?;
            let f = def.fields.iter().find(|f| f.name == *seg)?;
            ty = Some(f.field_type);
            if i + 1 < path.len() {
                match f.field_type {
                    HirType::Struct { name, .. } => cur = name,
                    _ => return None,
                }
            }
        }
        ty
    }

    /// Field paths (relative to `this`) that calling `func` may reallocate.
    /// Built from the callee's body on demand, so it doesn't matter whether or
    /// when the callee's module was checked.
    pub fn analyze_invalidations(
        &mut self,
        owner: StrId,
        func: &HirFunc<'a, 'bump>,
    ) -> Vec<Vec<StrId>> {
        let key = (owner, func.name);
        if let Some(c) = self.invalidation_cache.get(&key) {
            return c.clone();
        }
        self.invalidation_cache.insert(key, Vec::new()); // cycle guard
        let out = self.build_invalidations(owner, func);
        self.invalidation_cache.insert(key, out.clone());
        out
    }

    /// This computes the invalidated paths for a function and it depends on two sources:
    /// 1. Direct writes to `this.path`; assigning an owned pointer field frees the old allocation
    /// requiring the old allocation to be invalidated.
    /// Writing `this.path.len` will invalidate `this.path`,
    /// but element writes like data[index] do not change structure and therefore do not contradict the borrow checker.
    ///
    /// 2. Calls on `this` or `this.field` which can appear in reads as a path whose last segment isn't a field but a method name.
    /// Those method's invalidations are fetched recursively and cycles are cut by the cache guard, and re-rooted under the field prefix.
    ///
    /// Functions without a body or without a `this` param invalidate nothing.
    fn build_invalidations(&mut self, owner: StrId, func: &HirFunc<'a, 'bump>) -> Vec<Vec<StrId>> {
        let Some(body) = func.body else {
            return Vec::new();
        };
        let has_this = func
            .params
            .is_some_and(|ps| ps.iter().any(|p| matches!(p, HirParam::This { .. })));
        if !has_this {
            return Vec::new();
        }

        let mut writes = Vec::new();
        let mut reads = Vec::new();
        self.collect_root_accesses_stmt(&body, self.this_id, &mut writes, &mut reads);

        let mut out: Vec<Vec<StrId>> = Vec::new();

        // `this.<path> = ...` where the field is an owned pointer: the old
        // allocation dies (same rule check_assignment_expr uses in-function).
        for (path, index) in &writes {
            if index.is_some() {
                continue; // element write, not a structural change
            }
            // `this.<p>.len = ..` on an owned slice changes its extent: invalidates `<p>`.
            if path.len() >= 2 && path.last().is_some_and(|s| s.as_str() == "len") {
                let prefix = path[..path.len() - 1].to_vec(); // pretty much removes the "len" part, and then we check if it's an owned slice.
                if matches!(
                    self.field_type_at(owner, &prefix),
                    Some(ptr) if ptr.is_owned_slice()
                ) && !out.contains(&prefix)
                {
                    out.push(prefix);
                }
                continue;
            }
            if !path.is_empty()
                && matches!(
                    self.field_type_at(owner, path),
                    Some(HirType::OwnedPointer { .. })
                )
                && !out.contains(path)
            {
                out.push(path.clone());
            }
        }

        // `this.method()` / `this.field.method()` show up in `reads` as a path whose
        // last segment isn't a field. Pull in the callee's invalidations.
        for (path, index) in &reads {
            if index.is_some() {
                continue;
            }
            let Some((last, prefix)) = path.split_last() else {
                continue;
            };
            let holder = if prefix.is_empty() {
                owner
            } else {
                match self.field_type_at(owner, prefix) {
                    Some(HirType::Struct { name, .. }) => name,
                    _ => continue,
                }
            };
            if self
                .field_type_at(holder, std::slice::from_ref(last))
                .is_some()
            {
                continue; // plain field read
            }
            let Some(callee) = self
                .context
                .get_method(&str_id_to_string(holder), &last.to_string())
                .copied()
            else {
                continue;
            };
            for sub in self.analyze_invalidations(holder, &callee) {
                let mut full = prefix.to_vec();
                full.extend(sub);
                if !out.contains(&full) {
                    out.push(full);
                }
            }
        }
        out
    }

    /// At a method call: invalidates the places the callee may reallocate. The receiver's place is
    /// auto-dereferenced if it is a reference/pointer, each invalidated field path is projected onto
    /// it, and `reallocate_place` is called on the result.
    pub(crate) fn apply_call_invalidations(
        &mut self,
        receiver: &HirExpr<'a, 'bump>,
        owner: StrId,
        func: &HirFunc<'a, 'bump>,
    ) {
        let paths = self.analyze_invalidations(owner, func);
        if paths.is_empty() {
            return;
        }
        let Some(mut base) = self.resolve_place(receiver) else {
            return;
        };
        if matches!(
            self.peek_type(receiver),
            HirType::Ref { .. } | HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
        ) {
            base = self.borrow_checker.project_deref(base);
        }
        for path in paths {
            let mut place = base;
            for f in &path {
                place = self.borrow_checker.project_field(place, *f);
            }
            self.borrow_checker.reallocate_place(place);
        }
    }

    /// Fixed-point pass over the "function X calls `this.Y()`" graph (`fn_this_calls`): each caller
    /// inherits its callees' invalidated paths until nothing changes. Used for functions whose
    /// callees weren't analysed yet when they were first processed.
    pub fn propagate_invalidations(&mut self) {
        let mut callers_of: HashMap<StrId, Vec<StrId>> = HashMap::default();
        for (caller, callees) in &self.fn_this_calls {
            for callee in callees {
                callers_of.entry(*callee).or_default().push(*caller);
            }
        }

        let mut worklist: VecDeque<StrId> = self.fn_this_calls.keys().copied().collect();
        let mut in_queue: HashSet<StrId> = worklist.iter().copied().collect();

        while let Some(caller) = worklist.pop_front() {
            in_queue.remove(&caller);
            let callees = self.fn_this_calls.get(&caller).cloned().unwrap_or_default();
            for callee in callees {
                let paths = match self.fn_invalidates.get(&callee) {
                    Some(p) if !p.is_empty() => p.clone(),
                    _ => continue,
                };
                let entry = self.fn_invalidates.entry(caller).or_default();
                let prev_len = entry.len();
                for p in paths {
                    if !entry.contains(&p) {
                        entry.push(p);
                    }
                }
                if entry.len() != prev_len {
                    if let Some(recallers) = callers_of.get(&caller) {
                        for &rc in recallers {
                            if in_queue.insert(rc) {
                                worklist.push_back(rc);
                            }
                        }
                    }
                }
            }
        }
    }

    /// This checks if `local` is mentioned at `point` or if there is a CFG point reachable from it.
    pub fn local_used_after(&self, point: PointId, local: StrId) -> bool {
        self.liveness
            .get_or_init(|| self.build_liveness())
            .used_at_or_after(point, local)
    }

    /// This checks if `expr` mentions `local`
    pub fn expr_references_local(&self, expr: &HirExpr<'a, 'bump>, local: StrId) -> bool {
        let mut v = ReferencesLocal {
            target: local,
            found: false,
        };
        v.visit_expr(expr);
        v.found
    }

    /// This checks if `stmt` mentions `local`
    pub fn stmt_references_local(&self, stmt: &HirStmt<'a, 'bump>, local: StrId) -> bool {
        let mut v = ReferencesLocal {
            target: local,
            found: false,
        };
        v.visit_stmt(stmt);
        v.found
    }

    /// A statement walker that fills `point_locals_used`.
    ///
    /// Each statement that has a CFG point (`stmt_points`) updates `current_point` first,
    /// so every identifier found inside it (and inside nested expressions) is attributed to that statement's point.
    pub fn collect_locals_used_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        LocalsCollector { tc: self }.visit_stmt(stmt);
    }

    /// An expression walker that walks every `Ident` and adds them to the set of the current CFG point.
    pub fn collect_locals_used_expr(&mut self, expr: &HirExpr<'a, 'bump>) {
        LocalsCollector { tc: self }.visit_expr(expr);
    }
}

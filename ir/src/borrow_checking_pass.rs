use smallvec::SmallVec;

use crate::ir_hasher::HashSet;
use crate::{hir::StrId, ir_hasher::FxHashMap};

use crate::borrow_checker::{
    AliasReasoner, BorrowChecker, BorrowError, BorrowKind, BorrowResult, Bound, IndexContainer,
    Interval, Loan, LoanId, MemoryRelation, Place, PlaceId, PlaceInitStatus, Projection,
    Provenance, ProvenanceId, ProvenanceOrigin, Scope, StorageId,
};

const VERSION_OPAQUE_BASE: u32 = 0x8000_0000;

fn collect_symbols(b: &Bound, out: &mut SmallVec<StrId, 4>) {
    match b {
        Bound::Symbol(n) => {
            if !out.contains(n) {
                out.push(*n);
            }
        }
        Bound::Offset { base, .. } | Bound::Scale { base, .. } => collect_symbols(base, out),
        Bound::Sum(a, c) => {
            collect_symbols(a, out);
            collect_symbols(c, out);
        }
        Bound::Min(v) | Bound::Max(v) => v.iter().for_each(|x| collect_symbols(x, out)),
        Bound::Const(_) | Bound::SelfField(_) | Bound::FreshPerCall | Bound::Opaque(_) => {}
    }
}

fn substitute_symbols(b: &Bound, map: &FxHashMap<StrId, Bound>) -> Bound {
    match b {
        Bound::Symbol(n) => map.get(n).cloned().unwrap_or_else(|| b.clone()),
        Bound::Offset { base, offset } => Bound::Offset {
            base: Box::new(substitute_symbols(base, map)),
            offset: *offset,
        },
        Bound::Scale { base, factor } => Bound::Scale {
            base: Box::new(substitute_symbols(base, map)),
            factor: *factor,
        },
        Bound::Sum(a, c) => Bound::Sum(
            Box::new(substitute_symbols(a, map)),
            Box::new(substitute_symbols(c, map)),
        ),
        Bound::Min(v) => Bound::Min(v.iter().map(|x| substitute_symbols(x, map)).collect()),
        Bound::Max(v) => Bound::Max(v.iter().map(|x| substitute_symbols(x, map)).collect()),
        Bound::Const(_) | Bound::SelfField(_) | Bound::FreshPerCall | Bound::Opaque(_) => b.clone(),
    }
}

fn version_of(next: &mut u32, versions: &mut FxHashMap<StrId, Bound>, s: StrId) {
    versions.entry(s).or_insert_with(|| {
        let b = Bound::Opaque(*next);
        *next += 1;
        b
    });
}

impl BorrowChecker {
    pub fn new() -> Self {
        Self {
            next_place: PlaceId(0),
            next_loan: LoanId(0),
            next_provenance: ProvenanceId(0),

            places: Vec::default(),
            active_loans: Vec::default(),
            provenances: FxHashMap::default(),
            local_places: FxHashMap::default(),

            place_roots: FxHashMap::default(),
            root_loans: FxHashMap::default(),

            reasoning: AliasReasoner::new(),
            scopes: Vec::new(),
            place_to_local: FxHashMap::default(),
            place_provenance: FxHashMap::default(),
            field_places: FxHashMap::default(),
            deref_places: FxHashMap::default(),
            index_places: FxHashMap::default(),
            index_symbols: FxHashMap::default(),
            next_opaque: VERSION_OPAQUE_BASE,
            pointee_origin: FxHashMap::default(),
            place_init: FxHashMap::default(),
            storage_validity: FxHashMap::default(),
            place_storage: FxHashMap::default(),
            next_storage: StorageId(0),
            provenance_storage: FxHashMap::default(),
            has_invalidated_storage: false,
        }
    }

    /// Live-place lookup. Returns `None` for ids that are out of range or that were
    /// tombstoned by `end_scope` (their root entry is gone).
    #[inline]
    fn try_place(&self, id: PlaceId) -> Option<&Place> {
        if !self.place_roots.contains_key(&id) {
            return None;
        }
        self.places.get(id.0 as usize)
    }

    pub fn print_borrow_state(&self) {
        eprintln!("=== borrow checker state ===");

        eprintln!("-- scopes ({}) --", self.scopes.len());
        for (depth, scope) in self.scopes.iter().enumerate() {
            eprintln!(
                "  [{}] places={:?} loans={:?}",
                depth, scope.places, scope.loans
            );
        }

        eprintln!("-- locals --");
        for (name, place) in &self.local_places {
            eprintln!("  {:?} -> {:?}", name, place);
        }

        let live_places: Vec<&Place> = self
            .places
            .iter()
            .filter(|p| self.place_roots.contains_key(&p.id))
            .collect();

        eprintln!("-- places ({}) --", live_places.len());
        for place in live_places {
            let id = place.id;
            let root = self.place_roots.get(&id);
            let local = self.place_to_local.get(&id);

            let proj_str = match &place.projection {
                None => "root".to_string(),
                Some(Projection::Field(f)) => format!("field({:?})", f),
                Some(Projection::Deref) => "deref".to_string(),
                Some(Projection::Index { index, container }) => {
                    format!("index({:?}, {:?})", index, container)
                }
            };

            eprintln!(
                "  {:?}: parent={:?} proj={} root={:?} local={:?}",
                id, place.parent, proj_str, root, local
            );
        }

        let live_loans: Vec<&Loan> = self.active_loans.iter().flatten().collect();

        eprintln!("-- active loans ({}) --", live_loans.len());
        for loan in live_loans {
            eprintln!(
                "  {:?}: place={:?} kind={:?} provenance={:?} active={}",
                loan.id, loan.place, loan.kind, loan.provenance_id, loan.active
            );
        }

        eprintln!("-- root -> loans --");
        for (root, loans) in &self.root_loans {
            eprintln!("  {:?} -> {:?}", root, loans);
        }

        eprintln!("=============================");
    }

    fn ancestor_crosses_indirection(
        &self,
        ancestor: PlaceId,
        mut descendant: PlaceId,
    ) -> Option<bool> {
        if ancestor == descendant {
            return None;
        }
        let mut crossed = false;
        while let Some(place) = self.try_place(descendant) {
            if matches!(
                place.projection,
                Some(Projection::Index { .. }) | Some(Projection::Deref)
            ) {
                crossed = true;
            }
            match place.parent {
                Some(parent) if parent == ancestor => return Some(crossed),
                Some(parent) => descendant = parent,
                None => return None,
            }
        }
        None
    }

    fn shell_exempt(
        &self,
        place: PlaceId,
        kind: BorrowKind,
        loan_place: PlaceId,
        loan_kind: BorrowKind,
    ) -> bool {
        let non_exclusive = |k: BorrowKind| !matches!(k, BorrowKind::Mutable);
        if let Some(true) = self.ancestor_crosses_indirection(place, loan_place) {
            return non_exclusive(kind);
        }
        if let Some(true) = self.ancestor_crosses_indirection(loan_place, place) {
            return non_exclusive(loan_kind);
        }
        false
    }

    fn alloc_place_id(&mut self) -> PlaceId {
        let id = self.next_place;
        self.next_place.0 += 1;
        id
    }

    fn alloc_loan_id(&mut self) -> LoanId {
        let id = self.next_loan;
        self.next_loan.0 += 1;
        id
    }

    fn alloc_storage_id(&mut self) -> StorageId {
        let id = self.next_storage;
        self.next_storage.0 += 1;
        self.storage_validity.insert(id, true);
        id
    }

    fn alloc_provenance_id(&mut self) -> ProvenanceId {
        let id = self.next_provenance;
        self.next_provenance.0 += 1;
        id
    }

    /// Append a place. Ids are handed out sequentially by `alloc_place_id`, so the
    /// id must equal the next Vec index.
    fn push_place(&mut self, place: Place) {
        debug_assert_eq!(place.id.0 as usize, self.places.len());
        self.places.push(place);
    }

    pub fn borrow_shared(&mut self, place: PlaceId) -> BorrowResult<LoanId> {
        self.check_definite_conflict(place, BorrowKind::Shared)?;
        self.create_loan(place, BorrowKind::Shared)
    }

    pub fn borrow_alias(&mut self, place: PlaceId) -> BorrowResult<LoanId> {
        self.check_definite_conflict(place, BorrowKind::Alias)?;
        self.create_loan(place, BorrowKind::Alias)
    }

    pub fn borrow_mut(&mut self, place: PlaceId) -> BorrowResult<LoanId> {
        self.check_definite_conflict(place, BorrowKind::Mutable)?;
        self.create_loan(place, BorrowKind::Mutable)
    }

    fn check_definite_conflict(&self, place: PlaceId, kind: BorrowKind) -> BorrowResult<()> {
        let root = self.place_roots[&place];
        let Some(loans) = self.root_loans.get(&root) else {
            return Ok(());
        };

        for &loan_id in loans {
            let Some(loan) = self.loan(loan_id) else {
                continue;
            };

            if self.shell_exempt(place, kind, loan.place, loan.kind) {
                continue;
            }

            if let MemoryRelation::Overlap = self.overlaps(place, loan.place)? {
                if !kind.compatible_with(loan.kind) {
                    return Err(match (kind, loan.kind) {
                        (BorrowKind::Mutable, BorrowKind::Mutable) => {
                            BorrowError::AlreadyMutablyBorrowed { place: loan.place }
                        }
                        (BorrowKind::Mutable, _) | (_, BorrowKind::Mutable) => {
                            BorrowError::MutablyBorrowed { place: loan.place }
                        }
                        // remaining incompatible case: Alias vs Shared, either direction
                        _ => BorrowError::AliasConflict { place: loan.place },
                    });
                }
            }
        }
        Ok(())
    }

    pub fn check_use_shell(
        &self,
        place: PlaceId,
        kind: BorrowKind,
        access_provenance: Option<ProvenanceId>,
    ) -> BorrowResult<()> {
        if let Some(prov) = access_provenance {
            if !self.provenance_is_valid(prov) {
                return Err(BorrowError::InvalidatedReference { place });
            }
        }

        let root = self.place_roots[&place];
        let Some(loans) = self.root_loans.get(&root) else {
            return Ok(());
        };

        for &loan_id in loans {
            let Some(loan) = self.loan(loan_id) else {
                continue;
            };

            let through_this_loan = access_provenance == Some(loan.provenance_id)
                && (loan.place == place || self.place_is_ancestor(loan.place, place));

            if through_this_loan {
                continue;
            }

            if self.shell_exempt(place, kind, loan.place, loan.kind) {
                continue;
            }

            match self.overlaps(place, loan.place)? {
                MemoryRelation::Disjoint => {}

                MemoryRelation::Overlap | MemoryRelation::Unknown
                    if !kind.compatible_with(loan.kind) =>
                {
                    return Err(BorrowError::UnknownAlias {
                        lhs: place,
                        rhs: loan.place,
                    });
                }

                _ => {}
            }
        }

        Ok(())
    }

    pub fn check_use(
        &self,
        place: PlaceId,
        kind: BorrowKind,
        access_provenance: Option<ProvenanceId>,
    ) -> BorrowResult<()> {
        if let Some(prov) = access_provenance {
            if !self.provenance_is_valid(prov) {
                return Err(BorrowError::InvalidatedReference { place });
            }
        }

        match self.place_init_status(place) {
            PlaceInitStatus::Uninit => return Err(BorrowError::UseOfUninitialized { place }),
            PlaceInitStatus::Moved => return Err(BorrowError::UseAfterMove { place }),
            _ => {}
        }

        let root = self.place_roots[&place];
        let Some(loans) = self.root_loans.get(&root) else {
            return Ok(());
        };

        for &loan_id in loans {
            let Some(loan) = self.loan(loan_id) else {
                continue;
            };

            // A loan never conflicts with a use that passes THROUGH it: if
            // loan.place is `place` itself, or an ancestor of it (i.e. `place`
            // is a field/index/deref projection reached via the very
            // reference this loan backs), this IS the loan enabling the
            // access, not a competing borrow. Safe to skip unconditionally:
            // check_definite_conflict already forbids two active loans from
            // ever being in an ancestor/descendant relationship on the same
            // lineage, so at most one active loan can ever be an
            // ancestor-or-equal of `place`, it can't be masking a different,
            // genuinely conflicting loan.
            let through_this_loan = access_provenance == Some(loan.provenance_id)
                && (loan.place == place || self.place_is_ancestor(loan.place, place));

            if through_this_loan {
                continue;
            }

            match self.overlaps(place, loan.place)? {
                MemoryRelation::Disjoint => {}
                MemoryRelation::Overlap | MemoryRelation::Unknown
                    if !kind.compatible_with(loan.kind) =>
                {
                    return Err(BorrowError::UnknownAlias {
                        lhs: place,
                        rhs: loan.place,
                    });
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn place_is_ancestor(&self, ancestor: PlaceId, mut current: PlaceId) -> bool {
        while let Some(place) = self.try_place(current) {
            if place.id == ancestor {
                return true;
            }
            match place.parent {
                Some(parent) => current = parent,
                None => return false,
            }
        }
        false
    }

    fn create_loan(&mut self, place: PlaceId, kind: BorrowKind) -> BorrowResult<LoanId> {
        let loan_id = self.alloc_loan_id();
        let provenance_id = self.derive_provenance(place);

        let loan = Loan {
            id: loan_id,
            place,
            kind,
            provenance_id,
            active: true,
        };

        // Loan ids are sequential, so the id is the Vec index.
        debug_assert_eq!(loan_id.0 as usize, self.active_loans.len());
        self.active_loans.push(Some(loan));

        let root = self.place_roots[&place];
        self.root_loans.entry(root).or_default().push(loan_id);

        if let Some(scope) = self.scopes.last_mut() {
            scope.loans.push(loan_id);
        }

        Ok(loan_id)
    }

    /// Register a new local variable.
    pub fn declare_local(&mut self, local: StrId) -> PlaceId {
        let place_id = self.alloc_place_id();

        let storage = self.alloc_storage_id();

        self.push_place(Place {
            id: place_id,
            parent: None,
            projection: None,
            children: SmallVec::new(),
            root: place_id,
            depth: 1,
            storage: storage,
        });

        self.place_storage.insert(place_id, storage);
        if let Some(prev) = self.local_places.insert(local, place_id) {
            if let Some(scope) = self.scopes.last_mut() {
                scope.shadowed.push((local, prev));
            }
        }
        self.place_to_local.insert(place_id, local);

        // A local is its own root.
        self.place_roots.insert(place_id, place_id);

        if let Some(scope) = self.scopes.last_mut() {
            scope.places.push(place_id);
        }

        place_id
    }

    /// Called by the type checker once it has proven (or disproven) that
    /// `struct_name`'s indexing method yields non-overlapping memory for
    /// distinct indices.
    pub fn mark_disjoint_indexable(&mut self, struct_name: StrId, disjoint: bool) {
        self.reasoning
            .disjoint_indexable
            .insert(struct_name, disjoint);
    }

    pub fn project_index(
        &mut self,
        base: PlaceId,
        index: Interval,
        container: IndexContainer,
    ) -> PlaceId {
        let key = (base, index.lower.clone(), index.upper.clone());
        if let Some(&existing) = self.index_places.get(&key) {
            return existing;
        }

        let mut syms: SmallVec<StrId, 4> = SmallVec::new();
        collect_symbols(&index.lower, &mut syms);
        collect_symbols(&index.upper, &mut syms);

        let id = self.alloc_place_id();
        self.places[base.0 as usize].children.push(id);
        let root = self.place_roots[&base];
        let storage = self.place_storage[&base];
        self.push_place(Place {
            id,
            parent: Some(base),
            projection: Some(Projection::Index { index, container }),
            children: SmallVec::new(),
            root,
            depth: 1,
            storage,
        });
        self.place_storage.insert(id, storage);
        self.place_roots.insert(id, root);
        self.index_places.insert(key, id);
        for s in syms {
            self.index_symbols.entry(s).or_default().push(id);
        }
        id
    }

    pub fn begin_scope(&mut self) {
        self.scopes.push(Scope {
            loans: SmallVec::new(),
            places: SmallVec::new(),
            shadowed: SmallVec::new(),
            assumed_facts: Vec::new(),
            assumed_place_facts: Vec::new(),
            assumed_disjoint: Vec::new(),
            assumed_ranges: Vec::new(),
        });
    }

    /// Arithmetic facts assumed directly in the innermost scope (not in scopes nested in it).
    pub fn current_scope_facts(&self) -> Vec<(Bound, Bound, bool)> {
        // Facts killed since they were assumed (a variable they mention was reassigned) are
        // still in the scope's record; only report the ones that still hold.
        self.scopes
            .last()
            .map(|s| {
                s.assumed_facts
                    .iter()
                    .filter(|(l, r, eq)| {
                        let map = if *eq {
                            &self.reasoning.equal
                        } else {
                            &self.reasoning.not_equal
                        };
                        map.get(l).is_some_and(|set| set.contains(r))
                    })
                    .cloned()
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Called from `kill_facts_where` (so every reassignment / shadowing / address-taken
    /// event the type checker already reports goes through here) and from `end_scope`.
    fn version_symbols_where(&mut self, pred: &dyn Fn(StrId) -> bool) {
        let mut versions: FxHashMap<StrId, Bound> = FxHashMap::default();
        let mut next = self.next_opaque;

        // Index places.
        let hits: SmallVec<StrId, 4> = self
            .index_symbols
            .keys()
            .copied()
            .filter(|s| pred(*s))
            .collect();
        let mut affected: SmallVec<PlaceId, 8> = SmallVec::new();
        for s in hits {
            version_of(&mut next, &mut versions, s);
            if let Some(ids) = self.index_symbols.remove(&s) {
                for id in ids {
                    if !affected.contains(&id) {
                        affected.push(id);
                    }
                }
            }
        }
        for id in affected {
            if !self.place_roots.contains_key(&id) {
                continue; // dead place, stale entry
            }
            let place = &mut self.places[id.0 as usize];
            let Some(parent) = place.parent else { continue };
            let Some(Projection::Index { index, .. }) = &mut place.projection else {
                continue;
            };
            let old_key = (parent, index.lower.clone(), index.upper.clone());
            index.lower = substitute_symbols(&index.lower, &versions);
            index.upper = substitute_symbols(&index.upper, &versions);
            let new_key = (parent, index.lower.clone(), index.upper.clone());
            self.index_places.remove(&old_key);
            self.index_places.insert(new_key, id);
        }

        // Pointer pointee offsets (`record_pointee`).
        for (_, (_, iv)) in self.pointee_origin.iter_mut() {
            let mut syms: SmallVec<StrId, 4> = SmallVec::new();
            collect_symbols(&iv.lower, &mut syms);
            collect_symbols(&iv.upper, &mut syms);
            let mut changed = false;
            for s in syms {
                if pred(s) {
                    version_of(&mut next, &mut versions, s);
                    changed = true;
                }
            }
            if changed {
                iv.lower = substitute_symbols(&iv.lower, &versions);
                iv.upper = substitute_symbols(&iv.upper, &versions);
            }
        }

        self.next_opaque = next;
    }

    pub fn kill_facts_where(&mut self, pred: &dyn Fn(StrId) -> bool) {
        fn mentions(b: &Bound, pred: &dyn Fn(StrId) -> bool) -> bool {
            match b {
                Bound::Symbol(n) => pred(*n),
                Bound::Offset { base, .. } | Bound::Scale { base, .. } => mentions(base, pred),
                Bound::Sum(a, c) => mentions(a, pred) || mentions(c, pred),
                Bound::Min(v) | Bound::Max(v) => v.iter().any(|x| mentions(x, pred)),
                Bound::Const(_) | Bound::SelfField(_) | Bound::FreshPerCall | Bound::Opaque(_) => {
                    false
                }
            }
        }
        let iv_mentions = |i: &Interval| mentions(&i.lower, pred) || mentions(&i.upper, pred);

        for map in [
            &mut self.reasoning.equal,
            &mut self.reasoning.not_equal,
            &mut self.reasoning.less_than,
        ] {
            map.retain(|k, _| !mentions(k, pred));
            for set in map.values_mut() {
                set.retain(|b| !mentions(b, pred));
            }
        }
        self.reasoning
            .ranges
            .retain(|k, iv| !mentions(k, pred) && !iv_mentions(iv));
        self.reasoning
            .disjoint_intervals
            .retain(|(l, r)| !iv_mentions(l) && !iv_mentions(r));

        self.version_symbols_where(pred);
    }

    pub fn end_scope(&mut self) {
        let Some(scope) = self.scopes.pop() else {
            return;
        };

        for (lhs, rhs, is_equal) in &scope.assumed_facts {
            if *is_equal {
                self.reasoning.retract_equal(lhs, rhs);
            } else {
                self.reasoning.retract_not_equal(lhs, rhs);
            }
        }
        for (lhs, rhs) in &scope.assumed_place_facts {
            self.reasoning.retract_place_not_equal(lhs, rhs);
        }
        for (value, iv) in &scope.assumed_ranges {
            // Only retract our own entry; an inner kill or an outer overwrite stays as is.
            if self.reasoning.ranges.get(value) == Some(iv) {
                self.reasoning.ranges.remove(value);
            }
        }
        for (l, r) in &scope.assumed_disjoint {
            Self::remove_one(
                &mut self.reasoning.disjoint_intervals,
                &(l.clone(), r.clone()),
            );
            Self::remove_one(
                &mut self.reasoning.disjoint_intervals,
                &(r.clone(), l.clone()),
            );
        }

        for loan in scope.loans {
            self.end_loan(loan);
        }

        let mut dead_names: SmallVec<StrId, 4> = SmallVec::new();
        let mut work: SmallVec<PlaceId, 16> = SmallVec::new();
        for &root in &scope.places {
            work.push(root);
            while let Some(id) = work.pop() {
                self.release_place(id, &mut work);
            }
            if let Some(local) = self.place_to_local.remove(&root) {
                dead_names.push(local);
                if self.local_places.get(&local) == Some(&root) {
                    self.local_places.remove(&local);
                }
            }
        }

        for &(local, prev) in scope.shadowed.iter().rev() {
            if self.place_roots.contains_key(&prev) && !self.local_places.contains_key(&local) {
                self.local_places.insert(local, prev);
            }
        }

        if !dead_names.is_empty() && !self.index_symbols.is_empty() {
            self.version_symbols_where(&|s| dead_names.contains(&s));
        }
    }

    fn release_place(&mut self, id: PlaceId, work: &mut SmallVec<PlaceId, 16>) {
        let Some(place) = self.places.get_mut(id.0 as usize) else {
            return;
        };
        work.extend(std::mem::take(&mut place.children));

        match (place.parent, &place.projection) {
            (Some(p), Some(Projection::Field(f))) => {
                self.field_places.remove(&(p, *f));
            }
            (Some(p), Some(Projection::Deref)) => {
                self.deref_places.remove(&p);
            }
            (Some(p), Some(Projection::Index { index, .. })) => {
                self.index_places
                    .remove(&(p, index.lower.clone(), index.upper.clone()));
            }
            _ => {}
        }
        self.place_roots.remove(&id);
    }

    pub fn assume_places_not_equal_scoped(&mut self, lhs: PlaceId, rhs: PlaceId) {
        let already = self
            .reasoning
            .place_not_equal
            .get(&lhs)
            .is_some_and(|s| s.contains(&rhs));
        if already {
            return;
        }
        self.reasoning
            .place_not_equal
            .entry(lhs)
            .or_default()
            .insert(rhs);
        self.reasoning
            .place_not_equal
            .entry(rhs)
            .or_default()
            .insert(lhs);
        if let Some(scope) = self.scopes.last_mut() {
            scope.assumed_place_facts.push((lhs, rhs));
        }
    }

    pub fn end_loan_now(&mut self, loan: LoanId) {
        self.end_loan(loan);

        for scope in self.scopes.iter_mut() {
            scope.loans.retain(|l| *l != loan);
        }
    }

    pub fn loan_for_place(&self, place: PlaceId) -> Option<&LoanId> {
        let root = self.place_roots.get(&place)?;
        self.root_loans.get(root)?.last()
    }

    /// Child place via field projection.
    pub fn project_field(&mut self, base: PlaceId, field: StrId) -> PlaceId {
        if let Some(&existing) = self.field_places.get(&(base, field)) {
            return existing;
        }

        let id = self.alloc_place_id();
        self.places[base.0 as usize].children.push(id);

        let root = self.place_roots[&base];
        let storage = self.alloc_storage_id();

        self.push_place(Place {
            id,
            parent: Some(base),
            projection: Some(Projection::Field(field)),
            children: SmallVec::new(),
            root,
            depth: 1,
            storage,
        });

        self.place_roots.insert(id, root);
        self.place_storage.insert(id, storage);
        self.field_places.insert((base, field), id);

        id
    }

    /// Child place via dereference.
    pub fn project_deref(&mut self, base: PlaceId) -> PlaceId {
        if let Some(&existing) = self.deref_places.get(&base) {
            return existing;
        }

        let id = self.alloc_place_id();

        self.places[base.0 as usize].children.push(id);

        let root = self.place_roots[&base];
        let storage = self.place_storage[&base];

        self.push_place(Place {
            id,
            parent: Some(base),
            projection: Some(Projection::Deref),
            children: SmallVec::new(),
            root,
            depth: 1,
            storage,
        });

        self.place_roots.insert(id, root);
        self.place_storage.insert(id, storage);
        self.deref_places.insert(base, id);

        id
    }

    pub fn check_move(&self, place: PlaceId) -> BorrowResult<()> {
        match self.place_init_status(place) {
            PlaceInitStatus::Uninit => return Err(BorrowError::UseOfUninitialized { place }),
            PlaceInitStatus::Moved => return Err(BorrowError::UseAfterMove { place }),
            _ => {}
        }

        let root = self.place_roots[&place];
        let Some(loans) = self.root_loans.get(&root) else {
            return Ok(());
        };

        for &loan_id in loans {
            let Some(loan) = self.loan(loan_id) else {
                continue;
            };
            match self.overlaps(place, loan.place)? {
                MemoryRelation::Disjoint => {}
                MemoryRelation::Overlap | MemoryRelation::Unknown => {
                    return Err(BorrowError::CannotMoveBorrowed { place: loan.place });
                }
            }
        }
        Ok(())
    }

    pub fn end_loan(&mut self, loan: LoanId) {
        let Some(slot) = self.active_loans.get_mut(loan.0 as usize) else {
            return;
        };
        let Some(loan) = slot.take() else {
            return;
        };

        let root = self.place_roots[&loan.place];

        if let Some(loans) = self.root_loans.get_mut(&root) {
            loans.retain(|l| *l != loan.id);

            if loans.is_empty() {
                self.root_loans.remove(&root);
            }
        }
    }

    /// Provenance of a newly-created reference.
    pub fn derive_provenance(&mut self, place: PlaceId) -> ProvenanceId {
        let storage = self.place_storage[&place];
        let key = (place, storage);

        if let Some(&id) = self.place_provenance.get(&key) {
            return id;
        }

        let place_info = self.place(place).clone();

        let (origin, parents) = match (place_info.parent, place_info.projection) {
            (None, _) => (ProvenanceOrigin::Local { place, storage }, Vec::new()),

            (Some(parent), Some(Projection::Field(field))) => {
                let parent_prov = self.derive_provenance(parent);

                (
                    ProvenanceOrigin::Field {
                        parent: parent_prov,
                        field,
                    },
                    vec![parent_prov],
                )
            }

            (Some(parent), Some(Projection::Index { .. })) => {
                let parent_prov = self.derive_provenance(parent);

                (
                    ProvenanceOrigin::Index {
                        parent: parent_prov,
                    },
                    vec![parent_prov],
                )
            }

            (Some(parent), Some(Projection::Deref)) => {
                let parent_prov = self.derive_provenance(parent);

                (
                    ProvenanceOrigin::Deref {
                        parent: parent_prov,
                    },
                    vec![parent_prov],
                )
            }

            _ => unreachable!(),
        };

        let id = self.alloc_provenance_id();

        self.provenances.insert(
            id,
            Provenance {
                id,
                origin,
                parents,
            },
        );

        self.place_provenance.insert(key, id);
        self.provenance_storage.insert(id, storage);

        id
    }

    pub fn merge_provenance(&mut self, parents: &[ProvenanceId]) -> ProvenanceId {
        let mut unique = Vec::new();

        for &parent in parents {
            if !unique.contains(&parent) {
                unique.push(parent);
            }
        }

        if unique.len() == 1 {
            return unique[0];
        }

        let id = self.alloc_provenance_id();

        self.provenances.insert(
            id,
            Provenance {
                id,
                origin: ProvenanceOrigin::Merge,
                parents: unique,
            },
        );

        id
    }

    pub fn provenance_is_valid(&self, provenance: ProvenanceId) -> bool {
        if !self.has_invalidated_storage {
            return true;
        }

        let mut stack = vec![provenance];
        let mut visited = HashSet::default();

        while let Some(id) = stack.pop() {
            if !visited.insert(id) {
                continue;
            }

            let Some(prov) = self.provenances.get(&id) else {
                return false;
            };

            if let Some(&storage) = self.provenance_storage.get(&id) {
                if !self.storage_is_valid(storage) {
                    return false;
                }
            }

            stack.extend(prov.parents.iter().copied());
        }

        true
    }

    #[inline(always)]
    pub fn place(&self, id: PlaceId) -> &Place {
        &self.places[id.0 as usize]
    }

    #[inline(always)]
    pub fn loan(&self, id: LoanId) -> Option<&Loan> {
        self.active_loans.get(id.0 as usize)?.as_ref()
    }

    pub fn provenance(&self, id: ProvenanceId) -> Option<&Provenance> {
        self.provenances.get(&id)
    }

    pub fn local_place(&self, local: StrId) -> Option<&PlaceId> {
        self.local_places.get(&local)
    }

    pub fn invalidate_place(&mut self, place: PlaceId) {
        let old_storage = self.place_storage[&place];

        // Kill the old allocation generation.
        self.invalidate_storage(old_storage);

        // End active loans into the dead allocation.
        let mut stack = vec![place];
        let mut affected = HashSet::default();

        while let Some(current) = stack.pop() {
            if !affected.insert(current) {
                continue;
            }

            let children = self.places[current.0 as usize].children.clone();

            stack.extend(children);
        }

        let dead_loans: Vec<LoanId> = self
            .active_loans
            .iter()
            .flatten()
            .filter(|loan| affected.contains(&loan.place))
            .map(|loan| loan.id)
            .collect();

        for loan in dead_loans {
            self.end_loan(loan);
        }

        // Give the structural places a new allocation generation.
        let new_storage = self.alloc_storage_id();

        for current in affected {
            self.place_storage.insert(current, new_storage);
        }
    }

    pub fn kill_local_storage(&mut self, local: StrId) {
        let Some(&place) = self.local_places.get(&local) else {
            return;
        };
        if let Some(&s) = self.place_storage.get(&place) {
            self.invalidate_storage(s);
        }
    }

    pub fn root_local_of(&self, place: PlaceId) -> Option<StrId> {
        let root = self.place_roots.get(&place)?;
        self.place_to_local.get(root).copied()
    }

    pub fn reallocate_place(&mut self, place: PlaceId) {
        let old_storage = self.place_storage[&place];
        let new_storage = self.alloc_storage_id();

        self.invalidate_storage(old_storage);

        let mut stack = vec![place];

        while let Some(current) = stack.pop() {
            self.place_storage.insert(current, new_storage);

            let children = self.places[current.0 as usize].children.clone();

            stack.extend(children);
        }
    }

    pub fn overlaps(&self, lhs: PlaceId, rhs: PlaceId) -> BorrowResult<MemoryRelation> {
        if self.try_place(lhs).is_none() {
            return Err(BorrowError::PlaceNotFound(lhs));
        }

        if self.try_place(rhs).is_none() {
            return Err(BorrowError::PlaceNotFound(rhs));
        }

        Ok(self.reasoning.relation(&self.places, lhs, rhs))
    }

    pub fn assume_equal(&mut self, lhs: Bound, rhs: Bound) {
        self.assume_equal_scoped(lhs, rhs);
    }

    pub fn assume_not_equal(&mut self, lhs: Bound, rhs: Bound) {
        self.assume_not_equal_scoped(lhs, rhs);
    }

    fn insert_equal(&mut self, lhs: Bound, rhs: Bound) {
        self.reasoning
            .equal
            .entry(lhs.clone())
            .or_default()
            .insert(rhs.clone());
        self.reasoning.equal.entry(rhs).or_default().insert(lhs);
    }

    fn insert_not_equal(&mut self, lhs: Bound, rhs: Bound) {
        self.reasoning
            .not_equal
            .entry(lhs.clone())
            .or_default()
            .insert(rhs.clone());
        self.reasoning.not_equal.entry(rhs).or_default().insert(lhs);
    }

    pub fn assume_equal_scoped(&mut self, lhs: Bound, rhs: Bound) {
        let already = self
            .reasoning
            .equal
            .get(&lhs)
            .is_some_and(|s| s.contains(&rhs));
        self.insert_equal(lhs.clone(), rhs.clone());
        if already {
            return;
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.assumed_facts.push((lhs, rhs, true));
        }
    }

    pub fn assume_not_equal_scoped(&mut self, lhs: Bound, rhs: Bound) {
        let already = self
            .reasoning
            .not_equal
            .get(&lhs)
            .is_some_and(|s| s.contains(&rhs));
        self.insert_not_equal(lhs.clone(), rhs.clone());
        if already {
            return;
        }
        if let Some(scope) = self.scopes.last_mut() {
            scope.assumed_facts.push((lhs, rhs, false));
        }
    }

    pub fn assume_range(&mut self, value: Bound, lower: Bound, upper: Bound) {
        let iv = Interval { lower, upper };
        self.reasoning.ranges.insert(value.clone(), iv.clone());
        if let Some(scope) = self.scopes.last_mut() {
            scope.assumed_ranges.push((value, iv));
        }
    }

    pub fn assume_disjoint(&mut self, left: Interval, right: Interval) {
        self.reasoning
            .disjoint_intervals
            .push((left.clone(), right.clone()));
        self.reasoning
            .disjoint_intervals
            .push((right.clone(), left.clone()));
        if let Some(scope) = self.scopes.last_mut() {
            scope.assumed_disjoint.push((left, right));
        }
    }

    fn remove_one(v: &mut Vec<(Interval, Interval)>, x: &(Interval, Interval)) {
        if let Some(pos) = v.iter().rposition(|p| p == x) {
            v.swap_remove(pos);
        }
    }

    pub fn record_pointee(&mut self, ptr_place: PlaceId, base: PlaceId, offset: Interval) {
        self.pointee_origin.insert(ptr_place, (base, offset));
    }

    pub fn pointee_of(&self, ptr_place: PlaceId) -> Option<&(PlaceId, Interval)> {
        self.pointee_origin.get(&ptr_place)
    }

    pub fn mark_place_init(&mut self, place: PlaceId) {
        self.place_init.insert(place, PlaceInitStatus::Init);
    }

    pub fn mark_place_uninit(&mut self, place: PlaceId) {
        self.place_init.insert(place, PlaceInitStatus::Uninit);
    }

    pub fn mark_place_moved(&mut self, place: PlaceId) {
        self.place_init.insert(place, PlaceInitStatus::Moved);
    }

    pub fn place_init_status(&self, place: PlaceId) -> PlaceInitStatus {
        if let Some(status) = self.place_init.get(&place) {
            return *status;
        }
        if let Some(p) = self.try_place(place) {
            if let Some(parent) = p.parent {
                return self.place_init_status(parent);
            }
        }
        PlaceInitStatus::Init
    }

    pub fn is_less_than(&self, lhs: &Bound, rhs: &Bound) -> bool {
        if lhs == rhs {
            return false;
        }
        if let (Bound::Const(l), Bound::Const(r)) = (lhs, rhs) {
            return l < r;
        }
        self.reasoning
            .less_than
            .get(lhs)
            .map_or(false, |set| set.contains(rhs))
    }

    fn bound_is_pure(b: &Bound) -> bool {
        match b {
            Bound::Const(_) | Bound::Symbol(_) | Bound::Opaque(_) => true,
            Bound::Offset { base, .. } | Bound::Scale { base, .. } => Self::bound_is_pure(base),
            Bound::Sum(a, c) => Self::bound_is_pure(a) && Self::bound_is_pure(c),
            Bound::Min(v) | Bound::Max(v) => v.iter().all(Self::bound_is_pure),
            Bound::SelfField(_) | Bound::FreshPerCall => false,
        }
    }

    /// Splits `b` into `(base, constant)` so that `b == base + constant` (`base == None` means
    /// the bound is just a constant). Nested offsets are flattened. `None` if `b` isn't pure or
    /// the constants overflow, which makes every query below answer "can't prove".
    fn split_const_offset(b: &Bound) -> Option<(Option<Bound>, i64)> {
        match b {
            Bound::Const(c) => Some((None, *c)),
            Bound::Offset { base, offset } => {
                let (inner, o) = Self::split_const_offset(base)?;
                Some((inner, o.checked_add(*offset)?))
            }
            other if Self::bound_is_pure(other) => Some((Some(other.clone()), 0)),
            _ => None,
        }
    }

    /// Is `lhs ⋈ rhs` implied by one recorded fact `p ⋈ q` (where `⋈` is `==` or `!=`, the
    /// facts all being of the same kind)? Facts are symmetric, and shifting both sides of a fact
    /// by the same constant preserves it, so `i != j` also proves `i + 1 != j + 1`.
    fn implied_by_fact(
        lhs: &Bound,
        rhs: &Bound,
        facts: &FxHashMap<Bound, fxhash::FxHashSet<Bound>>,
    ) -> bool {
        let (Some((bl, ol)), Some((br, o_r))) =
            (Self::split_const_offset(lhs), Self::split_const_offset(rhs))
        else {
            return false;
        };
        let Some(diff) = ol.checked_sub(o_r) else {
            return false;
        };
        for (p, qs) in facts {
            let Some((bp, op)) = Self::split_const_offset(p) else {
                continue;
            };
            for q in qs {
                let Some((bq, oq)) = Self::split_const_offset(q) else {
                    continue;
                };
                let Some(fact_diff) = op.checked_sub(oq) else {
                    continue;
                };
                if bl == bp && br == bq && diff == fact_diff {
                    return true;
                }
                if bl == bq && br == bp && fact_diff.checked_neg() == Some(diff) {
                    return true;
                }
            }
        }
        false
    }

    pub fn proves_not_equal(&self, lhs: &Bound, rhs: &Bound) -> bool {
        if let (Some((bl, ol)), Some((br, o_r))) =
            (Self::split_const_offset(lhs), Self::split_const_offset(rhs))
        {
            if bl == br {
                return ol != o_r;
            }
        }
        Self::implied_by_fact(lhs, rhs, &self.reasoning.not_equal)
    }

    pub fn proves_equal(&self, lhs: &Bound, rhs: &Bound) -> bool {
        if let (Some((bl, ol)), Some((br, o_r))) =
            (Self::split_const_offset(lhs), Self::split_const_offset(rhs))
        {
            if bl == br && ol == o_r {
                return true;
            }
        }
        Self::implied_by_fact(lhs, rhs, &self.reasoning.equal)
    }

    pub fn proves_index_in_bounds(&self, index: &Bound, len: &Bound) -> bool {
        self.is_less_than(index, len)
    }

    pub fn proves_index_uninit(&self, index: &Bound, len: &Bound) -> bool {
        if index == len {
            return true;
        }
        self.is_less_than(len, index)
    }

    pub fn invalidate_storage(&mut self, storage: StorageId) {
        self.has_invalidated_storage = true; // This is safe because BorrowChecker is reset anyways.
        self.storage_validity.insert(storage, false);
    }

    pub fn storage_is_valid(&self, storage: StorageId) -> bool {
        self.storage_validity
            .get(&storage)
            .copied()
            .unwrap_or(false)
    }

    pub fn check_drop(&self, place: PlaceId) -> BorrowResult<()> {
        match self.place_init_status(place) {
            PlaceInitStatus::Uninit => Err(BorrowError::CannotDropUninitialized { place }),
            PlaceInitStatus::Moved => Err(BorrowError::UseAfterMove { place }),
            _ => Ok(()),
        }
    }

    pub fn reset(&mut self) {
        self.next_place = PlaceId(0);
        self.next_loan = LoanId(0);
        self.next_provenance = ProvenanceId(0);
        self.next_storage = StorageId(0);
        self.has_invalidated_storage = false;
        self.next_opaque = VERSION_OPAQUE_BASE;

        self.reasoning.reset();

        self.places.clear();
        self.active_loans.clear();
        self.provenances.clear();
        self.local_places.clear();
        self.place_roots.clear();
        self.root_loans.clear();
        self.scopes.clear();
        self.place_to_local.clear();
        self.place_provenance.clear();
        self.field_places.clear();
        self.deref_places.clear();
        self.index_places.clear();
        self.pointee_origin.clear();
        self.place_init.clear();
        self.storage_validity.clear();
        self.place_storage.clear();
        self.provenance_storage.clear();
        self.index_symbols.clear();
    }
}

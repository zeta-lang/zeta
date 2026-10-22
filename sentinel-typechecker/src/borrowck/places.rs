use ir::{
    borrow_checker::{
        BorrowError, BorrowKind, Bound, IndexContainer, Interval, LoanId, PlaceId, ProvenanceId,
    },
    errors::type_error::TypeErrorKind,
    hir::{
        HirEffectAccess, HirEffectSegment, HirExpr, HirTypeKind, Operator, ProvenanceAnnotation,
        RefKind,
    },
};

use crate::{TypeChecker, closures, naming::provenance_to_string, type_checker::LoanWatermark};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// The provenance of the reference that reading through `expr` goes via.
    /// Walks `*r`, `r.f`, `r[i]` back to the root local; only locals carry a provenance.
    pub fn access_provenance(&self, expr: &HirExpr) -> Option<ProvenanceId> {
        match expr {
            HirExpr::Ident(name, _) => self.local_provenance.get(name).copied(),

            HirExpr::Deref { expr, .. } => self.access_provenance(expr),

            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.access_provenance(object)
            }

            HirExpr::Index { object, .. } => self.access_provenance(object),

            _ => None,
        }
    }

    /// Maps an expression to the `PlaceId` it denotes, or `None` if it isn't a trackable place.
    ///
    /// The non-obvious cases:
    /// * **Field access through a pointer/reference** auto-derefs first (`p.f` is `(*p).f`).
    /// * **Owned slice/array pointers** (`OwnedPointer<[T]>`) are dereferenced before indexing.
    /// * **`arr[i]` / `slice[a..b]`** become an index projection over an interval; a single index
    ///   is the degenerate interval `[i, i]`.
    /// * **Raw/safe pointer arithmetic** (`ptr[i]`, `ptr + n`, `ptr - n`) does not name a fixed
    ///   place: it takes the pointer's current pointee `(base, offset)` and *adds* the new offset to
    ///   it, so `(p + 1)[2]` resolves to `base[offset + 3]`. Subtraction negates via `Scale(-1)`.
    pub fn resolve_place(&mut self, expr: &HirExpr<'a, 'bump>) -> Option<PlaceId> {
        match expr {
            HirExpr::Ident(name, _) => self
                .local_provenance_place
                .get(name)
                .copied()
                .or_else(|| self.borrow_checker.local_place(*name).copied()),

            HirExpr::This { .. } => self.borrow_checker.local_place(self.this_id).copied(),

            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let base = self.resolve_place(object)?;
                let base = match self.peek_type(object).kind {
                    HirTypeKind::Ref { .. }
                    | HirTypeKind::SafePointer { .. }
                    | HirTypeKind::UnsafePointer { .. }
                    | HirTypeKind::OwnedPointer { .. } => self.borrow_checker.project_deref(base),
                    _ => base,
                };
                Some(self.borrow_checker.project_field(base, *field))
            }

            HirExpr::Deref { expr, .. } => {
                let base = self.resolve_place(expr)?;
                Some(self.borrow_checker.project_deref(base))
            }

            HirExpr::Slice {
                object,
                start,
                end,
                inclusive,
                ..
            } => match self.peek_type(object).kind {
                HirTypeKind::Array(_, _) | HirTypeKind::Slice(_) => {
                    let base = self.resolve_place(object)?;
                    let interval = self.slice_interval(start, end, *inclusive);
                    Some(self.borrow_checker.project_index(
                        base,
                        interval,
                        IndexContainer::Primitive,
                    ))
                }
                HirTypeKind::OwnedPointer { inner, .. }
                    if matches!(inner.kind, HirTypeKind::Slice(_) | HirTypeKind::Array(_, _)) =>
                {
                    let base = self.resolve_place(object)?;
                    let deref_base = self.borrow_checker.project_deref(base);
                    let interval = self.slice_interval(start, end, *inclusive);
                    Some(self.borrow_checker.project_index(
                        deref_base,
                        interval,
                        IndexContainer::Primitive,
                    ))
                }
                _ => None,
            },

            HirExpr::Index { object, index, .. } => match self.peek_type(object).kind {
                HirTypeKind::Array(_, _) | HirTypeKind::Slice(_) => {
                    let base = self.resolve_place(object)?;
                    let bound = self.expr_to_bound(index);
                    let interval = Interval {
                        lower: bound.clone(),
                        upper: bound,
                    };
                    Some(self.borrow_checker.project_index(
                        base,
                        interval,
                        IndexContainer::Primitive,
                    ))
                }
                HirTypeKind::OwnedPointer { inner, .. }
                    if matches!(inner.kind, HirTypeKind::Slice(_) | HirTypeKind::Array(_, _)) =>
                {
                    let base = self.resolve_place(object)?;
                    let deref_base = self.borrow_checker.project_deref(base);
                    let bound = self.expr_to_bound(index);
                    let interval = Interval {
                        lower: bound.clone(),
                        upper: bound,
                    };
                    Some(self.borrow_checker.project_index(
                        deref_base,
                        interval,
                        IndexContainer::Primitive,
                    ))
                }
                HirTypeKind::SafePointer { .. } | HirTypeKind::UnsafePointer { .. } => {
                    let ptr_place = self.resolve_place(object)?;
                    let (base, cur) = self.borrow_checker.pointee_of(ptr_place)?.clone();
                    let idx_bound = self.expr_to_bound(index);
                    let combined = Bound::Sum(Box::new(cur.lower.clone()), Box::new(idx_bound));
                    let interval = Interval {
                        lower: combined.clone(),
                        upper: combined,
                    };
                    Some(self.borrow_checker.project_index(
                        base,
                        interval,
                        IndexContainer::Primitive,
                    ))
                }
                _ => None,
            },

            HirExpr::Binary {
                left,
                op: op @ (Operator::Add | Operator::Subtract),
                right,
                ..
            } => {
                if !matches!(
                    self.peek_type(left).kind,
                    HirTypeKind::SafePointer { .. } | HirTypeKind::UnsafePointer { .. }
                ) {
                    return None;
                }
                let ptr_place = self.resolve_place(left)?;
                let (base, cur) = self.borrow_checker.pointee_of(ptr_place)?.clone();
                let delta = self.expr_to_bound(right);
                let signed = match op {
                    Operator::Subtract => Bound::Scale {
                        base: Box::new(delta),
                        factor: -1,
                    },
                    _ => delta,
                };
                let combined = Bound::Sum(Box::new(cur.lower.clone()), Box::new(signed));
                let interval = Interval {
                    lower: combined.clone(),
                    upper: combined,
                };
                Some(
                    self.borrow_checker
                        .project_index(base, interval, IndexContainer::Primitive),
                )
            }

            _ => None,
        }
    }

    /// If `expr` is a local that *owns a loan* (it holds a reference), get the place that loan borrows.
    pub fn loan_referent_place(&self, expr: &HirExpr<'a, 'bump>) -> Option<PlaceId> {
        let HirExpr::Ident(name, _) = expr else {
            return None;
        };
        let loan_id = self
            .loan_owners
            .iter()
            .find(|(_, owner)| **owner == *name)
            .map(|(&id, _)| id)?;
        self.borrow_checker.loan(loan_id).map(|loan| loan.place)
    }

    /// True if `place` is `ancestor` or lies underneath it (walks parent links upward).
    pub fn place_is_within(&self, place: PlaceId, ancestor: PlaceId) -> bool {
        let mut cur = Some(place);
        while let Some(p) = cur {
            if p == ancestor {
                return true;
            }
            cur = self.borrow_checker.place(p).parent;
        }
        false
    }

    /// A fresh unknown bound. Every call gets a distinct id, so two unknown indices are never
    /// treated as equal (or provably disjoint) by the overlap analysis.
    pub fn fresh_opaque(&mut self) -> Bound {
        self.next_opaque_id += 1;
        Bound::Opaque(self.next_opaque_id)
    }

    /// Evaluates integer constants, including constant locals and checked `+ - * / %`.
    /// Returns `None` on overflow, division by zero, or anything non-constant.
    pub fn const_eval(&self, expr: &HirExpr<'a, 'bump>) -> Option<i64> {
        match expr {
            HirExpr::Number(v, _) => Some(*v),
            HirExpr::Ident(name, _) => {
                let (sym, _) = self.context.get_variable(name)?;
                self.const_symbols.get(&sym).copied()
            }
            HirExpr::Binary {
                left, op, right, ..
            } => {
                let l = self.const_eval(left)?;
                let r = self.const_eval(right)?;
                match op {
                    Operator::Add => l.checked_add(r),
                    Operator::Subtract => l.checked_sub(r),
                    Operator::Multiply => l.checked_mul(r),
                    Operator::Divide if r != 0 => l.checked_div(r),
                    Operator::Modulo if r != 0 => l.checked_rem(r),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Turns an index expression into a symbolic `Bound` the checker can reason about:
    /// constants fold to `Const`, a variable is a `Symbol`, `x + c` is `Offset`, `x * c` is `Scale`,
    /// and anything else (including `x + y` of two unknowns) becomes a fresh `Opaque`.
    pub fn expr_to_bound(&mut self, expr: &HirExpr<'a, 'bump>) -> Bound {
        if let Some(c) = self.const_eval(expr) {
            return Bound::Const(c);
        }
        match expr {
            HirExpr::Number(value, _) => Bound::Const(*value),
            HirExpr::Ident(name, _) => Bound::Symbol(*name),

            HirExpr::Binary {
                left,
                op: Operator::Add,
                right,
                ..
            } => match (self.expr_to_bound(left), self.expr_to_bound(right)) {
                (base, Bound::Const(c)) | (Bound::Const(c), base) => Bound::Offset {
                    base: Box::new(base),
                    offset: c,
                },
                _ => self.fresh_opaque(),
            },

            HirExpr::Binary {
                left,
                op: Operator::Subtract,
                right,
                ..
            } => match (self.expr_to_bound(left), self.expr_to_bound(right)) {
                (base, Bound::Const(c)) => Bound::Offset {
                    base: Box::new(base),
                    offset: -c,
                },
                _ => self.fresh_opaque(),
            },

            HirExpr::Binary {
                left,
                op: Operator::Multiply,
                right,
                ..
            } => match (self.expr_to_bound(left), self.expr_to_bound(right)) {
                (base, Bound::Const(c)) | (Bound::Const(c), base) => Bound::Scale {
                    base: Box::new(base),
                    factor: c,
                },
                _ => self.fresh_opaque(),
            },

            _ => self.fresh_opaque(),
        }
    }

    /// Index interval of `start..end` / `start..=end`. Intervals are half-open, so an inclusive
    /// end `b` becomes `b + 1`.
    pub fn slice_interval(
        &mut self,
        start: &HirExpr<'a, 'bump>,
        end: &HirExpr<'a, 'bump>,
        inclusive: bool,
    ) -> Interval {
        let lower = self.expr_to_bound(start);
        let mut upper = self.expr_to_bound(end);
        if inclusive {
            // half-open: `a..=b` is `[a, b+1)`
            upper = match upper {
                Bound::Const(c) => Bound::Const(c + 1),
                Bound::Offset { base, offset } => Bound::Offset {
                    base,
                    offset: offset + 1,
                },
                other => Bound::Offset {
                    base: Box::new(other),
                    offset: 1,
                },
            };
        }
        Interval { lower, upper }
    }

    /// `ref_a == ref_b` / `ref_a != ref_b` between two reference locals, as
    /// `(place_a, place_b, is_equal)`: lets the borrow checker learn whether two references point
    /// at the same place inside a branch.
    pub fn condition_to_place_fact(
        &self,
        cond: &HirExpr<'a, 'bump>,
    ) -> Option<(PlaceId, PlaceId, bool)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };
        let is_equal = match op {
            Operator::Equals => true,
            Operator::NotEquals => false,
            _ => return None,
        };
        let lp = self.loan_referent_place(left)?;
        let rp = self.loan_referent_place(right)?;
        Some((lp, rp, is_equal))
    }

    /// Same idea for index expressions: `i == j` / `i != j` as `(bound_i, bound_j, is_equal)`.
    pub fn condition_to_fact(&mut self, cond: &HirExpr<'a, 'bump>) -> Option<(Bound, Bound, bool)> {
        if let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        {
            let is_equal = match op {
                Operator::Equals => true,
                Operator::NotEquals => false,
                _ => return None,
            };
            Some((
                self.expr_to_bound(left),
                self.expr_to_bound(right),
                is_equal,
            ))
        } else {
            None
        }
    }

    /// Checks that using `place` with `kind` doesn't conflict with active loans, and that the
    /// reference we are going through hasn't been invalidated. Also notes the use for closure
    /// capture analysis.
    pub fn check_borrow_use(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        place: PlaceId,
        kind: BorrowKind,
    ) {
        if let Some((root, path)) = self.effect_path_of(expr) {
            self.note_capture_use(root, &path, closures::UseLevel::from(kind));
        }

        let access_provenance = self.access_provenance(expr);
        self.check_provenance_still_valid(access_provenance);

        if let Err(e) = self
            .borrow_checker
            .check_use(place, kind, access_provenance)
        {
            let provenance = self.infer_provenance(expr);
            let msg = self.describe_borrow_error(&e, provenance.as_ref());
            self.record(TypeErrorKind::Generic(msg));
        }
    }

    /// Identical to [`Self::check_borrow_use`] except it calls the borrow checker's
    /// `check_use_shell` variant.
    /// `_shell` allows mutable references to contents to co-exist with immutable references to what holds those contents, allowing `list.get_mut(i)` to co-exist with `list.len()`
    pub fn check_borrow_use_shell(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        place: PlaceId,
        kind: BorrowKind,
    ) {
        if let Some((root, path)) = self.effect_path_of(expr) {
            self.note_capture_use(root, &path, closures::UseLevel::from(kind));
        }
        let access_provenance = self.access_provenance(expr);
        self.check_provenance_still_valid(access_provenance);

        if let Err(e) = self
            .borrow_checker
            .check_use_shell(place, kind, access_provenance)
        {
            let provenance = self.infer_provenance(expr);
            let msg = self.describe_borrow_error(&e, provenance.as_ref());
            self.record(TypeErrorKind::Generic(msg));
        }
    }

    pub fn describe_borrow_error(
        &self,
        err: &BorrowError,
        provenance: Option<&ProvenanceAnnotation>,
    ) -> String {
        let base = match err {
            BorrowError::UseAfterMove { .. } => "use of a value after it was moved".to_string(),
            BorrowError::InvalidatedReference { .. } => {
                "the reference's lifetime has ended because it was invalidated".to_string()
            }
            BorrowError::MutablyBorrowed { .. } => {
                "cannot borrow: value is already mutably borrowed".to_string()
            }
            BorrowError::AlreadyMutablyBorrowed { .. } => {
                "cannot borrow as mutable: already mutably borrowed elsewhere".to_string()
            }
            BorrowError::Borrowed { .. } => {
                "cannot borrow as mutable: value is already borrowed".to_string()
            }
            BorrowError::InvalidMove { .. } => "invalid move".to_string(),
            BorrowError::InvalidWrite { .. } => "invalid write".to_string(),
            BorrowError::InvalidRead { .. } => "invalid read".to_string(),
            BorrowError::CannotMoveBorrowed { .. } => {
                "cannot move out of a value while it is borrowed".to_string()
            }
            BorrowError::UnknownAlias { .. } => {
                "cannot prove these two accesses don't overlap".to_string()
            }
            BorrowError::UseOfUninitialized { .. } => "use of uninitialized value".to_string(),
            BorrowError::CannotDropUninitialized { .. } => {
                "cannot drop uninitialized value".to_string()
            }
            BorrowError::LoanNotFound(_)
            | BorrowError::PlaceNotFound(_)
            | BorrowError::ProvenanceNotFound(_) => "internal borrow-checker error".to_string(),

            BorrowError::AliasConflict { .. } => {
                "cannot borrow as &alias, an immutable reference or a mutable reference co-exists."
                    .to_string()
            }
        };

        match provenance {
            Some(p) => format!(
                "{} (via {})",
                base,
                provenance_to_string(p, &self.context.string_pool)
            ),
            None => base.to_string(),
        }
    }

    pub fn in_unsafe(&self) -> bool {
        self.unsafe_depth != 0
    }

    /// For a callee whose effect annotation lists several accessed paths, borrows each place
    /// (following `Field` / `Index` segments) with the annotated ref kind. Returns the loans taken;
    /// borrow errors are recorded rather than returned.
    pub fn register_multi_place_loans(
        &mut self,
        base_expr: &HirExpr<'a, 'bump>,
        accesses: &[HirEffectAccess<'bump>],
    ) -> Vec<LoanId> {
        let mut loans = Vec::new();
        let Some(base_place) = self.resolve_place(base_expr) else {
            return loans;
        };
        for access in accesses.iter() {
            let mut pid = base_place;
            for seg in access.path.iter() {
                pid = match seg {
                    HirEffectSegment::Field(f) => self.borrow_checker.project_field(pid, *f),
                    HirEffectSegment::Index(key) => {
                        let bound = self.effect_index_key_to_bound(key);
                        let interval = Interval {
                            lower: bound.clone(),
                            upper: bound,
                        };
                        self.borrow_checker
                            .project_index(pid, interval, IndexContainer::Primitive)
                    }
                };
            }
            let result = match access.ref_kind {
                RefKind::Shared => self.borrow_checker.borrow_shared(pid),
                RefKind::Alias => self.borrow_checker.borrow_alias(pid),
                RefKind::Unique => self.borrow_checker.borrow_mut(pid),
            };
            match result {
                Ok(loan_id) => loans.push(loan_id),
                Err(e) => self.record(TypeErrorKind::Generic(self.describe_borrow_error(&e, None))),
            }
        }
        loans
    }

    /// Loans created while checking one expression are *temporaries*: snapshot the live keys
    /// before, then end every loan whose key is new afterwards.
    pub fn snapshot_temp_loans(&self) -> LoanWatermark {
        LoanWatermark {
            call: self.call_loan_order.len(),
            closure: self.closure_loan_order.len(),
        }
    }

    pub fn end_temp_call_loans(&mut self, watermark: LoanWatermark) {
        while self.call_loan_order.len() > watermark.call {
            let (key, loan_id) = self.call_loan_order.pop().unwrap();

            if self.call_loans.get(&key) == Some(&loan_id) {
                self.call_loans.remove(&key);
                self.borrow_checker.end_loan_now(loan_id);
            }
        }
    }

    pub fn end_temp_closure_loans(&mut self, watermark: LoanWatermark) {
        while self.closure_loan_order.len() > watermark.closure {
            let (key, loan_id) = self.closure_loan_order.pop().unwrap();

            let removed = if let Some(ids) = self.closure_loans.get_mut(&key) {
                if let Some(index) = ids.iter().position(|&id| id == loan_id) {
                    ids.swap_remove(index);
                    true
                } else {
                    false
                }
            } else {
                false
            };

            if removed {
                self.borrow_checker.end_loan_now(loan_id);
            }

            if self
                .closure_loans
                .get(&key)
                .is_some_and(|ids| ids.is_empty())
            {
                self.closure_loans.remove(&key);
            }
        }
    }
}

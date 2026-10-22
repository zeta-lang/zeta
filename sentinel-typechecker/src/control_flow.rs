use std::rc::Rc;

use ir::{
    borrow_checker::{BorrowKind, LoanId, PlaceId},
    errors::type_error::TypeErrorKind,
    hir::{
        HirExpr, HirMatchArm, HirPattern, HirPatternKind, HirStmt, HirType, HirTypeKind,
        ProvenanceAnnotation, RefKind, StrId,
    },
    ir_hasher::{FxHashBuilder, HashSet},
    span::SourceSpan,
};

use crate::{
    TypeChecker,
    initialization::BindingMode,
    move_state::MoveState,
    naming::type_to_string,
    type_checker::{InvalidationMap, NonNullState, Occurrence, SymbolId},
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// Union `from` into `into` without copying when nothing changed.
    /// Both maps are copy-on-write, so the common "this branch invalidated nothing"
    /// case is a pointer comparison.
    fn merge_invalidations(into: &mut Rc<InvalidationMap>, from: Rc<InvalidationMap>) {
        if Rc::ptr_eq(into, &from) || from.is_empty() {
            return;
        }
        if into.is_empty() {
            *into = from;
            return;
        }
        let dst = Rc::make_mut(into);
        match Rc::try_unwrap(from) {
            Ok(owned) => dst.extend(owned),
            Err(shared) => dst.extend(shared.iter().map(|(k, v)| (*k, *v))),
        }
    }

    pub fn check_match_arms(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        let scrutinee_ty = self.check_expr(expr);
        let scrutinee_ty = match scrutinee_ty.kind {
            HirTypeKind::Ref { inner, .. } => *inner,
            _ => scrutinee_ty,
        };

        self.check_match_exhaustiveness(&scrutinee_ty, arms);

        let scrutinee_loan = self.take_scrutinee_loan(expr);

        let move_state_before = self.move_state.clone();
        let mut arm_types = Vec::with_capacity(arms.len());
        let mut arm_move_states = Vec::with_capacity(arms.len());

        let inval_before = Rc::clone(&self.invalidated_provenances);
        let mut inval_after = Rc::clone(&inval_before);
        for arm in arms {
            self.invalidated_provenances = Rc::clone(&inval_before);
            let (arm_type, arm_move_state) = self.check_match_arm(
                expr,
                arm,
                &scrutinee_ty,
                expected,
                &move_state_before,
                scrutinee_loan,
            );
            let diverges = matches!(arm_type.kind, HirTypeKind::Never);
            if !diverges {
                let arm_inval = std::mem::take(&mut self.invalidated_provenances);
                Self::merge_invalidations(&mut inval_after, arm_inval);
            }

            arm_types.push(arm_type);
            if !diverges {
                arm_move_states.push(arm_move_state);
            }
        }

        self.invalidated_provenances = inval_after;

        let mut joined = move_state_before;
        for state in &arm_move_states {
            joined.join_assign(state);
        }
        self.move_state = joined;

        self.join_branch_types(&arm_types, expected)
    }

    fn take_scrutinee_loan(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
    ) -> Option<(PlaceId, BorrowKind)> {
        let id = self.call_loans.remove(&Self::expr_key(scrutinee))?;
        let info = self.borrow_checker.loan(id).map(|l| (l.place, l.kind));
        self.borrow_checker.end_loan_now(id);
        info
    }

    pub fn check_match_arm(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        arm: &HirMatchArm<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
        move_state_before: &MoveState,
        scrutinee_loan: Option<(PlaceId, BorrowKind)>,
    ) -> (HirType<'a, 'bump>, MoveState) {
        let binding_mode = self.scrutinee_binding_mode(scrutinee);
        let scrutinee_place = self.resolve_place(scrutinee);
        let scrutinee_provenance = self.infer_provenance(scrutinee);

        self.binding_mode_backfill
            .insert(Self::expr_key(scrutinee), binding_mode);

        self.move_state.clone_from(move_state_before);

        self.borrow_checker.begin_scope();

        let arm_context = self.context.create_child_scope();
        let old_context = std::mem::replace(&mut self.context, arm_context);

        self.check_match_arm_pattern(
            arm,
            scrutinee_ty,
            binding_mode,
            scrutinee_provenance,
            scrutinee_place,
        );

        let holders =
            self.bind_arm_provenance(&arm.pattern, scrutinee_ty, binding_mode, scrutinee_loan);

        self.check_match_arm_guard(arm.guard);

        let arm_type = self.check_arm_body(arm.body, expected, &holders);

        self.context = old_context;
        self.borrow_checker.end_scope();
        if !holders.is_empty() {
            self.loan_owners.retain(|_, owner| !holders.contains(owner));
        }

        self.cleanup_match_arm_bindings(&arm.pattern, scrutinee_ty);
        // The caller either resets `move_state` for the next arm or overwrites it with the
        // join, so the end-state can be moved out instead of cloned.
        (arm_type, std::mem::take(&mut self.move_state))
    }

    fn check_arm_body(
        &mut self,
        body: &HirStmt<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
        holders: &[StrId],
    ) -> HirType<'a, 'bump> {
        let HirStmt::Block { body, span: _ } = body else {
            unreachable!()
        };
        if !holders.is_empty() && !self.loan_owners.is_empty() {
            let dead: Vec<LoanId> = self
                .loan_owners
                .iter()
                .filter(|(_, o)| holders.contains(o))
                .filter(|(_, o)| !body.iter().any(|s| self.stmt_references_local(s, **o)))
                .map(|(&id, _)| id)
                .collect();
            for id in dead {
                self.borrow_checker.end_loan_now(id);
                self.loan_owners.remove(&id);
            }
        }
        let Some((last, rest)) = body.split_last() else {
            return HirType::synthetic(HirTypeKind::Void);
        };
        for (i, s) in rest.iter().enumerate() {
            self.check_stmt(s);
            self.end_dead_block_loans(body, i, holders);
        }
        self.check_tail_expected(last, expected)
    }

    pub fn check_match_arm_pattern(
        &mut self,
        arm: &HirMatchArm<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        binding_mode: BindingMode,
        scrutinee_provenance: Option<ProvenanceAnnotation<'bump>>,
        scrutinee_place: Option<PlaceId>,
    ) {
        self.check_pattern_against_type(&arm.pattern, scrutinee_ty);

        self.register_pattern_bindings(
            &arm.pattern,
            scrutinee_ty,
            binding_mode,
            scrutinee_provenance,
            scrutinee_place,
        );
    }

    pub fn check_match_arm_guard(&mut self, guard: Option<&HirExpr<'a, 'bump>>) {
        let Some(guard) = guard else {
            return;
        };

        let guard_type = self.check_expr(guard);

        if guard_type.kind != HirTypeKind::Boolean {
            self.record(TypeErrorKind::TypeMismatch {
                expected: "bool".to_string(),
                found: type_to_string(&guard_type, &self.context.string_pool).to_string(),
            });
        }
    }

    pub fn cleanup_match_arm_bindings(
        &mut self,
        pattern: &HirPattern<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
    ) {
        let mut bound = Vec::new();
        self.collect_pattern_bindings(pattern, scrutinee_ty, &mut bound);

        for (name, _) in bound {
            self.local_provenance.remove(&name);
            self.local_provenance_place.remove(&name);
            self.local_ref_kind.remove(&name);
        }
    }

    /// Returns the bindings that hold the re-created scrutinee loan.
    pub fn bind_arm_provenance(
        &mut self,
        pattern: &HirPattern<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        mode: BindingMode,
        loan: Option<(PlaceId, BorrowKind)>,
    ) -> Vec<StrId> {
        let mut holders = Vec::new();
        if !matches!(mode, BindingMode::ByValue) {
            return holders; // ByRef bindings get their own loan in register_pattern_bindings
        }
        let Some((place, kind)) = loan else {
            return holders;
        };

        let mut bound = Vec::new();
        self.collect_pattern_bindings(pattern, scrutinee_ty, &mut bound);
        let refs: Vec<(StrId, HirType<'a, 'bump>)> = bound
            .into_iter()
            .filter(|(_, ty)| {
                matches!(
                    ty.kind,
                    HirTypeKind::Ref { .. }
                        | HirTypeKind::SafePointer { .. }
                        | HirTypeKind::UnsafePointer { .. }
                )
            })
            .collect();
        if refs.is_empty() {
            return holders; // e.g. `case null`: no reference escapes, no loan
        }

        let result = match kind {
            BorrowKind::Mutable => self.borrow_checker.borrow_mut(place),
            BorrowKind::Alias => self.borrow_checker.borrow_alias(place),
            BorrowKind::Shared => self.borrow_checker.borrow_shared(place),
        };
        let loan_id = match result {
            Ok(id) => id,
            Err(e) => {
                let msg = self.describe_borrow_error(&e, None);
                self.record(TypeErrorKind::Generic(msg));
                return holders;
            }
        };
        let Some(prov) = self.borrow_checker.loan(loan_id).map(|l| l.provenance_id) else {
            return holders;
        };

        for (name, ty) in refs {
            self.local_provenance.insert(name, prov);
            self.local_provenance_place.insert(name, place);
            if let HirTypeKind::Ref { ref_kind, .. } = ty.kind {
                self.local_ref_kind.insert(name, ref_kind);
            }
            holders.push(name);
        }
        self.loan_owners.insert(loan_id, holders[0]);
        holders
    }

    pub fn check_block_body(
        &mut self,
        body: &[HirStmt<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        self.borrow_checker.begin_scope();

        let mut block_context = self.context.create_child_scope();
        let local_names = self.collect_block_local_names(body);

        let mut value = None;
        let last_idx = body.len().checked_sub(1);

        for (i, stmt) in body.iter().enumerate() {
            std::mem::swap(&mut self.context, &mut block_context);

            value = if Some(i) == last_idx {
                Some(self.check_tail_expected(stmt, expected))
            } else {
                self.check_stmt(stmt)
            };

            std::mem::swap(&mut self.context, &mut block_context);

            self.end_dead_block_loans(body, i, &local_names);
        }

        self.borrow_checker.end_scope();
        self.cleanup_block_locals(&local_names);

        value
    }

    pub fn collect_block_local_names(&self, body: &[HirStmt<'a, 'bump>]) -> Vec<StrId> {
        body.iter()
            .filter_map(|stmt| match stmt {
                HirStmt::Let { name, .. } => Some(*name),
                _ => None,
            })
            .collect()
    }

    pub fn end_dead_block_loans(
        &mut self,
        body: &[HirStmt<'a, 'bump>],
        statement_index: usize,
        local_names: &[StrId],
    ) {
        // Only loans owned by one of `local_names` can die here; skip the scan entirely
        // when there are none.
        if self.loan_owners.is_empty() || local_names.is_empty() {
            return;
        }

        let after_point = self
            .stmt_after_points
            .get(&Self::stmt_key(&body[statement_index]))
            .copied();

        let dead_loans: Vec<LoanId> = self
            .loan_owners
            .iter()
            .filter(|(_, owner)| local_names.contains(owner))
            .filter(|(_, owner)| match after_point {
                Some(point) => !self.local_used_after(point, **owner),
                None => !body[(statement_index + 1)..]
                    .iter()
                    .any(|stmt| self.stmt_references_local(stmt, **owner)),
            })
            .map(|(&loan_id, _)| loan_id)
            .collect();

        for loan_id in dead_loans {
            self.borrow_checker.end_loan_now(loan_id);
            self.loan_owners.remove(&loan_id);
        }
    }

    pub fn cleanup_block_locals(&mut self, local_names: &[StrId]) {
        if !self.loan_owners.is_empty() {
            self.loan_owners
                .retain(|_, owner| !local_names.contains(owner));
        }

        for name in local_names {
            self.local_provenance_place.remove(name);
            self.local_ref_kind.remove(name);
        }
    }

    /// The `else` path starts from the pre-`if` state (already swapped in by the caller);
    /// only the path-sensitive non-null refinement for the negated condition is applied here.
    /// The arithmetic / place facts live in the borrow checker's scope stack, which is not
    /// path-sensitive, so they are assumed inside the `else` scope (and after the `if`, when
    /// `then` diverges).
    fn restore_non_null_for_else(&mut self, cond: &HirExpr<'a, 'bump>) {
        if let Some((root, path, holds_true)) = self.condition_to_non_null_fact(cond) {
            if !holds_true {
                self.mark_non_null(root, &path);
            }
        }
    }

    pub fn check_if_branches(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&HirStmt<'a, 'bump>>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        self.check_if_condition(cond);

        // `non_null_state` and `invalidated_provenances` are copy-on-write: these are O(1)
        // snapshots, and a branch only pays for a copy if it actually mutates them.
        let move_state_before = self.move_state.clone();
        let inval_before = Rc::clone(&self.invalidated_provenances);
        let non_null_before = Rc::clone(&self.non_null_state);

        let then_value = self.check_if_then_branch(cond, then_block, expected);
        let then_diverges = matches!(then_value.kind, HirTypeKind::Never);

        // Take the `then` end-state and install the `else` start-state in one step.
        let then_move_state = std::mem::replace(&mut self.move_state, move_state_before);
        let then_non_null = std::mem::replace(&mut self.non_null_state, non_null_before);
        let then_inval = std::mem::replace(&mut self.invalidated_provenances, inval_before);

        self.restore_non_null_for_else(cond);

        let else_value = self.check_if_else_branch(cond, else_block, expected);
        let else_inval = std::mem::take(&mut self.invalidated_provenances);
        let else_diverges = else_value
            .as_ref()
            .is_some_and(|t| matches!(t.kind, HirTypeKind::Never));
        self.invalidated_provenances = match (then_diverges, else_diverges) {
            (true, false) => else_inval,
            (false, true) => then_inval,
            _ => {
                let mut u = then_inval;
                Self::merge_invalidations(&mut u, else_inval);
                u
            }
        };
        let else_move_state = std::mem::take(&mut self.move_state);
        let else_non_null = std::mem::take(&mut self.non_null_state);

        self.join_if_branch_states(
            then_move_state,
            else_move_state,
            then_non_null,
            else_non_null,
            then_diverges,
            else_diverges,
        );

        // Each branch's condition facts were only assumed inside that branch's own scope. Past
        // the `if`, a fact survives only when the *other* branch cannot fall through: after
        // `if i == j { return }` we are on the `i != j` path, but after `if i != j { .. }`
        // (no `else`, or a `then` that falls through) we know nothing.
        match (then_diverges, else_diverges) {
            (true, false) => self.assume_if_condition_is_false(cond),
            (false, true) => self.assume_if_condition_is_true(cond),
            _ => {}
        }

        else_value.map(|else_type| self.join_branch_types(&[then_value, else_type], expected))
    }

    pub fn check_if_condition(&mut self, cond: &HirExpr<'a, 'bump>) {
        let snap = self.snapshot_temp_loans();
        let cond_type = self.check_expr(cond);
        // Calls that certainly ran to produce the condition hold their `ensures` in both
        // branches and after the statement.
        self.apply_unconditional_call_facts(cond);
        self.end_temp_call_loans(snap);
        self.end_temp_closure_loans(snap);

        if cond_type.kind != HirTypeKind::Boolean {
            self.record(TypeErrorKind::TypeMismatch {
                expected: "bool".to_string(),
                found: type_to_string(&cond_type, &self.context.string_pool).to_string(),
            });
        }
    }

    pub fn check_if_then_branch(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        self.borrow_checker.begin_scope();
        self.assume_if_condition_is_true(cond);

        let mut then_context = self.context.create_child_scope();
        let mut then_value = HirType::synthetic(HirTypeKind::Void);

        if let Some((last, rest)) = then_block.split_last() {
            for stmt in rest {
                std::mem::swap(&mut self.context, &mut then_context);

                self.check_stmt(stmt);

                std::mem::swap(&mut self.context, &mut then_context);
            }

            std::mem::swap(&mut self.context, &mut then_context);

            then_value = self.check_tail_expected(last, expected);

            std::mem::swap(&mut self.context, &mut then_context);
        }

        self.borrow_checker.end_scope();

        then_value
    }

    pub fn assume_if_condition_is_true(&mut self, cond: &HirExpr<'a, 'bump>) {
        self.apply_path_call_facts(cond, true);
        let fact = self.condition_to_fact(cond);
        let non_null_fact = self.condition_to_non_null_fact(cond);
        let place_fact = self.condition_to_place_fact(cond);

        if let Some((lhs, rhs, is_equal)) = fact
            .as_ref()
            .filter(|(l, r, _)| self.fact_is_trackable(l, r))
        {
            if *is_equal {
                self.borrow_checker
                    .assume_equal_scoped(lhs.clone(), rhs.clone());
            } else {
                self.borrow_checker
                    .assume_not_equal_scoped(lhs.clone(), rhs.clone());
            }
        }

        if let Some((lhs, rhs, is_equal)) = place_fact {
            if !is_equal {
                self.borrow_checker.assume_places_not_equal_scoped(lhs, rhs);
            }
        }

        if let Some((root, path, holds_true)) = &non_null_fact {
            if *holds_true {
                self.mark_non_null(*root, path);
            }
        }
    }

    pub fn assume_if_condition_is_false(&mut self, cond: &HirExpr<'a, 'bump>) {
        self.apply_path_call_facts(cond, false);
        let fact = self.condition_to_fact(cond);
        let non_null_fact = self.condition_to_non_null_fact(cond);
        let place_fact = self.condition_to_place_fact(cond);

        if let Some((lhs, rhs, is_equal)) = fact
            .as_ref()
            .filter(|(l, r, _)| self.fact_is_trackable(l, r))
        {
            if *is_equal {
                self.borrow_checker
                    .assume_not_equal_scoped(lhs.clone(), rhs.clone());
            } else {
                self.borrow_checker
                    .assume_equal_scoped(lhs.clone(), rhs.clone());
            }
        }

        if let Some((lhs, rhs, is_equal)) = place_fact {
            if is_equal {
                self.borrow_checker.assume_places_not_equal_scoped(lhs, rhs);
            }
        }

        if let Some((root, path, holds_true)) = &non_null_fact {
            if !*holds_true {
                self.mark_non_null(*root, path);
            }
        }
    }

    pub fn check_if_else_branch(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        else_block: Option<&HirStmt<'a, 'bump>>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        let else_stmt = else_block?;

        self.borrow_checker.begin_scope();
        self.assume_if_condition_is_false(cond);

        let else_context = self.context.create_child_scope();
        let old_context = std::mem::replace(&mut self.context, else_context);

        let else_value = self.check_tail_expected(else_stmt, expected);

        self.context = old_context;
        self.borrow_checker.end_scope();

        Some(else_value)
    }

    pub fn join_if_branch_states(
        &mut self,
        then_move_state: MoveState,
        else_move_state: MoveState,
        then_non_null: NonNullState,
        else_non_null: NonNullState,
        then_diverges: bool,
        else_diverges: bool,
    ) {
        self.move_state = match (then_diverges, else_diverges) {
            (true, false) => else_move_state,
            (false, true) => then_move_state,
            _ => {
                let mut joined = then_move_state;
                joined.join_assign(&else_move_state);
                joined
            }
        };

        self.non_null_state = match (then_diverges, else_diverges) {
            (true, false) => else_non_null,
            (false, true) => then_non_null,
            // Neither branch touched the non-null facts: nothing to join.
            _ if Rc::ptr_eq(&then_non_null, &else_non_null) => then_non_null,
            _ => Rc::new(Self::join_non_null_states(&then_non_null, &else_non_null)),
        };
    }

    /// `(field name, field type)` of `variant` on the enum type `enum_ty`, with the enum's generic
    /// arguments substituted (`Result<u8, E>` + `Ok` => `[(value, u8)]`).
    fn enum_variant_fields(
        &self,
        enum_ty: &HirType<'a, 'bump>,
        variant: StrId,
    ) -> Option<Vec<(StrId, HirType<'a, 'bump>)>> {
        let HirTypeKind::Enum {
            name, type_args, ..
        } = &enum_ty.kind
        else {
            return None;
        };
        let def = self.context.get_enum(name)?;
        let variant_def = def.variants.iter().find(|v| v.name == variant)?;
        let subs = self.generic_substitutions_for_type(*name, type_args);
        Some(
            variant_def
                .fields
                .iter()
                .map(|f| {
                    let ty = if subs.is_empty() {
                        f.field_type
                    } else {
                        self.substitute_type_local(&f.field_type, &subs)
                    };
                    (f.name, ty)
                })
                .collect(),
        )
    }

    /// A pattern binding is a fresh variable: forget any move recorded for an earlier
    /// binding that happened to share its name.
    fn reset_binding_move_state(&mut self, name: StrId) {
        if let Some(local) = self.local_table.get(name) {
            self.move_state.clear(local);
        }
    }

    pub fn register_pattern_bindings(
        &mut self,
        pattern: &HirPattern<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        mode: BindingMode,
        scrutinee_provenance: Option<ProvenanceAnnotation<'bump>>,
        scrutinee_place: Option<PlaceId>,
    ) {
        if let HirTypeKind::Nullable(inner) = &scrutinee_ty.kind {
            if !matches!(pattern.kind, HirPatternKind::Null) {
                let inner_ty = **inner;
                return self.register_pattern_bindings(
                    pattern,
                    &inner_ty,
                    mode,
                    scrutinee_provenance,
                    scrutinee_place,
                );
            }
        }

        match pattern.kind {
            HirPatternKind::Ident(name, is_mut) => {
                let symbol_id = self.mint_symbol_id();
                let bound_ty = self.bind_leaf_type(*scrutinee_ty, mode, scrutinee_provenance);
                if self.context.get_variable(name).is_some() {
                    self.record(TypeErrorKind::VariableAlreadyExists {
                        var_name: name.to_string(),
                    });
                }
                self.context
                    .add_variable_with_mutability(name, bound_ty, is_mut, symbol_id);
                self.reset_binding_move_state(name);
                self.record_binding_occurrence(pattern.span, name, bound_ty, symbol_id);

                self.borrow_checker.declare_local(name);
                self.kill_facts_about(name);

                match mode {
                    BindingMode::ByRef(rk) => {
                        self.local_ref_kind.insert(name, rk);
                        if let Some(src_place) = scrutinee_place {
                            self.local_provenance_place.insert(name, src_place);
                            let result = match rk {
                                RefKind::Unique => self.borrow_checker.borrow_mut(src_place),
                                RefKind::Alias => self.borrow_checker.borrow_alias(src_place),
                                RefKind::Shared => self.borrow_checker.borrow_shared(src_place),
                            };
                            match result {
                                Ok(loan_id) => {
                                    self.loan_owners.insert(loan_id, name);
                                }
                                Err(e) => {
                                    let msg = self
                                        .describe_borrow_error(&e, scrutinee_provenance.as_ref());
                                    self.record(TypeErrorKind::Generic(msg));
                                }
                            }
                        }
                    }
                    BindingMode::ByValue => {
                        // owned
                    }
                }
            }

            HirPatternKind::EnumVariant {
                variant,
                bindings,
                variant_span: _,
            } => {
                let Some(variant_fields) = self.enum_variant_fields(scrutinee_ty, variant) else {
                    return;
                };
                for (binding_name, (field_name, field_ty)) in bindings.iter().zip(variant_fields) {
                    let symbol_id = self.mint_symbol_id();
                    let field_provenance =
                        Self::extend_provenance(scrutinee_provenance, field_name, &self.context);
                    let bound_ty = self.bind_leaf_type(field_ty, mode, field_provenance);
                    if self.context.get_variable(binding_name).is_some() {
                        self.record(TypeErrorKind::VariableAlreadyExists {
                            var_name: binding_name.to_string(),
                        });
                    }
                    self.context.add_variable(binding_name, bound_ty, symbol_id);
                    self.reset_binding_move_state(*binding_name);
                    self.record_binding_occurrence(
                        pattern.span,
                        *binding_name,
                        bound_ty,
                        symbol_id,
                    );
                }
            }

            HirPatternKind::Tuple(patterns) => {
                if let HirTypeKind::Tuple(elems) = &scrutinee_ty.kind {
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.register_pattern_bindings(
                            sub_pattern,
                            elem_ty,
                            mode,
                            scrutinee_provenance,
                            scrutinee_place,
                        );
                    }
                }
            }

            HirPatternKind::Array(patterns) => {
                let elem_ty = match &scrutinee_ty.kind {
                    HirTypeKind::Array(inner, _) | HirTypeKind::Slice(inner) => Some(**inner),
                    _ => None,
                };
                if let Some(elem_ty) = elem_ty {
                    for sub_pattern in patterns.iter() {
                        self.register_pattern_bindings(
                            sub_pattern,
                            &elem_ty,
                            mode,
                            scrutinee_provenance,
                            scrutinee_place,
                        );
                    }
                }
            }

            HirPatternKind::Struct {
                name,
                fields,
                variant_span: _,
            } => match &scrutinee_ty.kind {
                HirTypeKind::Enum { .. } => {
                    let Some(variant_fields) = self.enum_variant_fields(scrutinee_ty, name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(&(_, field_ty)) =
                            variant_fields.iter().find(|(n, _)| n == field_name)
                        else {
                            continue;
                        };
                        let field_provenance = Self::extend_provenance(
                            scrutinee_provenance,
                            *field_name,
                            &self.context,
                        );
                        self.register_pattern_bindings(
                            sub_pattern,
                            &field_ty,
                            mode,
                            field_provenance,
                            scrutinee_place,
                        );
                    }
                }
                HirTypeKind::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let Some(def) = self.context.get_struct(struct_name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        let field_ty = field_types
                            .get(field_idx)
                            .copied()
                            .unwrap_or(def.fields[field_idx].field_type);
                        let field_provenance = Self::extend_provenance(
                            scrutinee_provenance,
                            *field_name,
                            &self.context,
                        );
                        self.register_pattern_bindings(
                            sub_pattern,
                            &field_ty,
                            mode,
                            field_provenance,
                            scrutinee_place,
                        );
                    }
                }
                _ => {}
            },

            HirPatternKind::Or(patterns) => {
                if let Some(first) = patterns.first() {
                    self.register_pattern_bindings(
                        first,
                        scrutinee_ty,
                        mode,
                        scrutinee_provenance,
                        scrutinee_place,
                    );
                }
            }

            _ => {}
        }
    }

    fn record_binding_occurrence(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        ty: HirType<'a, 'bump>,
        symbol: SymbolId,
    ) {
        self.occurrences.push(Occurrence {
            span,
            name,
            hir_name: name,
            ty,
            module_idx: self.context.current_module_idx,
            symbol,
            is_decl: true,
        });
    }

    pub fn bind_leaf_type(
        &self,
        field_ty: HirType<'a, 'bump>,
        mode: BindingMode,
        provenance: Option<ProvenanceAnnotation<'bump>>,
    ) -> HirType<'a, 'bump> {
        match mode {
            BindingMode::ByValue => field_ty,
            BindingMode::ByRef(rk) => HirType::synthetic(HirTypeKind::Ref {
                inner: self.context.bump.alloc_value(field_ty),
                ref_kind: rk,
                provenance,
            }),
        }
    }

    pub fn collect_pattern_bindings(
        &self,
        pattern: &HirPattern<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        out: &mut Vec<(StrId, HirType<'a, 'bump>)>,
    ) {
        if let HirTypeKind::Nullable(inner) = &scrutinee_ty.kind {
            if !matches!(pattern.kind, HirPatternKind::Null) {
                let inner_ty = **inner;
                return self.collect_pattern_bindings(pattern, &inner_ty, out);
            }
        }

        match pattern.kind {
            HirPatternKind::Ident(name, _) => {
                out.push((name, *scrutinee_ty));
            }

            HirPatternKind::EnumVariant {
                variant,
                bindings,
                variant_span: _,
            } => {
                let Some(variant_fields) = self.enum_variant_fields(scrutinee_ty, variant) else {
                    return;
                };
                for (binding_name, (_, field_ty)) in bindings.iter().zip(variant_fields) {
                    out.push((*binding_name, field_ty));
                }
            }

            HirPatternKind::Tuple(patterns) => {
                if let HirTypeKind::Tuple(elems) = &scrutinee_ty.kind {
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.collect_pattern_bindings(sub_pattern, elem_ty, out);
                    }
                }
            }

            HirPatternKind::Array(patterns) => {
                let elem_ty = match &scrutinee_ty.kind {
                    HirTypeKind::Array(inner, _) | HirTypeKind::Slice(inner) => Some(**inner),
                    _ => None,
                };
                if let Some(elem_ty) = elem_ty {
                    for sub_pattern in patterns.iter() {
                        self.collect_pattern_bindings(sub_pattern, &elem_ty, out);
                    }
                }
            }

            HirPatternKind::Struct {
                name,
                fields,
                variant_span: _,
            } => match &scrutinee_ty.kind {
                HirTypeKind::Enum { .. } => {
                    let Some(variant_fields) = self.enum_variant_fields(scrutinee_ty, name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(&(_, field_ty)) =
                            variant_fields.iter().find(|(n, _)| n == field_name)
                        else {
                            continue;
                        };
                        self.collect_pattern_bindings(sub_pattern, &field_ty, out);
                    }
                }
                HirTypeKind::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let Some(def) = self.context.get_struct(struct_name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        let field_ty = field_types
                            .get(field_idx)
                            .copied()
                            .unwrap_or(def.fields[field_idx].field_type);
                        self.collect_pattern_bindings(sub_pattern, &field_ty, out);
                    }
                }
                _ => {}
            },

            HirPatternKind::Or(patterns) => {
                if let Some(first) = patterns.first() {
                    self.collect_pattern_bindings(first, scrutinee_ty, out);
                }
            }

            HirPatternKind::Wildcard
            | HirPatternKind::Number(_)
            | HirPatternKind::String(_)
            | HirPatternKind::Boolean(_)
            | HirPatternKind::Null => {}
        }
    }

    pub fn bindings_match(
        &self,
        a: &[(StrId, HirType<'a, 'bump>)],
        b: &[(StrId, HirType<'a, 'bump>)],
    ) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|((na, ta), (nb, tb))| na == nb && self.types_structurally_equal(ta, tb))
    }

    fn record_variant_pattern_occurrences(
        &mut self,
        pattern_span: SourceSpan<'a>,
        scrutinee_ty: &HirType<'a, 'bump>,
        variant: StrId,
        variant_span: SourceSpan<'a>,
    ) {
        if variant_span.line == 0 {
            return;
        }
        let HirTypeKind::Enum {
            name: enum_name, ..
        } = &scrutinee_ty.kind
        else {
            return;
        };
        let Some(def) = self.context.get_enum(enum_name) else {
            return;
        };
        if !def.variants.iter().any(|v| v.name == variant) {
            return;
        }

        self.occurrences.push(Occurrence {
            span: variant_span,
            name: variant,
            hir_name: variant,
            ty: *scrutinee_ty,
            module_idx: self.context.current_module_idx,
            symbol: SymbolId::Variant {
                enum_name: *enum_name,
                variant_name: variant,
            },
            is_decl: false,
        });

        // Only a qualified pattern has an enum name in front of the variant.
        let qualified =
            pattern_span.line == variant_span.line && pattern_span.column < variant_span.column;
        if !qualified {
            return;
        }
        if let (Some(owner), Some(name_span)) = (
            self.context.enum_owner(enum_name.as_str()),
            Self::enum_name_span_before_variant(variant_span, def.unmangled_name),
        ) {
            self.record_item_occurrence_named(
                name_span,
                def.unmangled_name,
                def.name,
                *scrutinee_ty,
                owner,
            );
        }
    }

    pub fn check_pattern_against_type(
        &mut self,
        pattern: &HirPattern<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
    ) {
        if let HirTypeKind::Nullable(inner) = &scrutinee_ty.kind {
            if !matches!(pattern.kind, HirPatternKind::Null) {
                let inner_ty = **inner;
                return self.check_pattern_against_type(pattern, &inner_ty);
            }
        }
        match pattern.kind {
            HirPatternKind::Null => {
                if !matches!(scrutinee_ty.kind, HirTypeKind::Nullable(_)) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: "null".to_string(),
                    });
                }
            }
            HirPatternKind::EnumVariant {
                variant,
                bindings,
                variant_span,
            } => {
                let HirTypeKind::Enum {
                    name: enum_name, ..
                } = &scrutinee_ty.kind
                else {
                    return;
                };
                let Some(def) = self.context.get_enum(enum_name) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == variant) else {
                    self.record(TypeErrorKind::Generic(format!(
                        "enum `{}` has no variant `{}`",
                        enum_name, variant
                    )));
                    return;
                };
                self.record_variant_pattern_occurrences(
                    pattern.span,
                    scrutinee_ty,
                    variant,
                    variant_span,
                );
                if bindings.len() != variant_def.fields.len() {
                    self.record(TypeErrorKind::Generic(format!(
                        "variant `{}::{}` has {} field(s), but the pattern binds {}",
                        enum_name,
                        variant,
                        variant_def.fields.len(),
                        bindings.len()
                    )));
                }
            }
            HirPatternKind::Boolean(_) => {
                if !matches!(scrutinee_ty.kind, HirTypeKind::Boolean) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: "bool".to_string(),
                    });
                }
            }
            HirPatternKind::Number(_) => {
                if !self.is_integer(scrutinee_ty) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: "integer".to_string(),
                    });
                }
            }
            HirPatternKind::String(_) => {
                if !matches!(scrutinee_ty.kind, HirTypeKind::String) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: "str".to_string(),
                    });
                }
            }

            HirPatternKind::Ident(_, _) | HirPatternKind::Wildcard => {}

            HirPatternKind::Tuple(patterns) => match &scrutinee_ty.kind {
                HirTypeKind::Tuple(elems) => {
                    if patterns.len() != elems.len() {
                        self.record(TypeErrorKind::Generic(format!(
                            "tuple pattern has {} element(s), but the scrutinee has {}",
                            patterns.len(),
                            elems.len()
                        )));
                    }
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.check_pattern_against_type(sub_pattern, elem_ty);
                    }
                }
                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: format!("tuple pattern with {} element(s)", patterns.len()),
                    });
                }
            },

            HirPatternKind::Array(patterns) => match &scrutinee_ty.kind {
                HirTypeKind::Array(inner, len) => {
                    if patterns.len() != *len {
                        self.record(TypeErrorKind::Generic(format!(
                            "array pattern has {} element(s), but the array type `{}` has length {}",
                            patterns.len(),
                            type_to_string(scrutinee_ty, &self.context.string_pool).to_string(),
                            len
                        )));
                    }
                    for sub_pattern in patterns.iter() {
                        self.check_pattern_against_type(sub_pattern, inner);
                    }
                }
                HirTypeKind::Slice(inner) => {
                    for sub_pattern in patterns.iter() {
                        self.check_pattern_against_type(sub_pattern, inner);
                    }
                }
                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: format!("array pattern with {} element(s)", patterns.len()),
                    });
                }
            },

            HirPatternKind::Struct {
                name,
                fields,
                variant_span,
            } => match &scrutinee_ty.kind {
                HirTypeKind::Enum {
                    name: enum_name, ..
                } => {
                    let Some(def) = self.context.get_enum(enum_name) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == name) else {
                        self.record(TypeErrorKind::Generic(format!(
                            "enum `{}` has no variant `{}`",
                            enum_name, name
                        )));
                        return;
                    };
                    self.record_variant_pattern_occurrences(
                        pattern.span,
                        scrutinee_ty,
                        name,
                        variant_span,
                    );

                    let mut seen: HashSet<StrId> = HashSet::default();
                    let variant_fields = self
                        .enum_variant_fields(scrutinee_ty, name)
                        .unwrap_or_default();
                    for (field_name, sub_pattern) in fields.iter() {
                        if !seen.insert(*field_name) {
                            self.record(TypeErrorKind::Generic(format!(
                                "field `{}` matched more than once in this pattern",
                                field_name
                            )));
                            continue;
                        }
                        let Some(idx) = variant_def
                            .fields
                            .iter()
                            .position(|f| f.name == *field_name)
                        else {
                            self.record(TypeErrorKind::Generic(format!(
                                "variant `{}::{}` has no field `{}`",
                                enum_name, name, field_name
                            )));
                            continue;
                        };
                        let field_ty = variant_fields[idx].1;
                        self.check_pattern_against_type(sub_pattern, &field_ty);
                    }

                    let missing: Vec<&str> = variant_def
                        .fields
                        .iter()
                        .filter(|f| !fields.iter().any(|(fname, _)| fname == &f.name))
                        .map(|f| f.name.as_str())
                        .collect();
                    if !missing.is_empty() {
                        self.record(TypeErrorKind::Generic(format!(
                            "pattern doesn't bind field(s) {} of variant `{}::{}`",
                            missing.join(", "),
                            enum_name,
                            name
                        )));
                    }
                }

                HirTypeKind::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let Some(def) = self.context.get_struct(struct_name) else {
                        return;
                    };

                    let mut seen: std::collections::HashSet<StrId> =
                        std::collections::HashSet::new();
                    for (field_name, sub_pattern) in fields.iter() {
                        if !seen.insert(*field_name) {
                            self.record(TypeErrorKind::Generic(format!(
                                "field `{}` matched more than once in this pattern",
                                field_name
                            )));
                            continue;
                        }
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            self.record(TypeErrorKind::FieldNotFound {
                                struct_name: struct_name.to_string(),
                                field: field_name.to_string(),
                            });
                            continue;
                        };
                        let field_ty = field_types
                            .get(field_idx)
                            .copied()
                            .unwrap_or(def.fields[field_idx].field_type);
                        self.check_pattern_against_type(sub_pattern, &field_ty);
                    }
                }

                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty, &self.context.string_pool)
                            .to_string(),
                        found: format!("named-field pattern `{}`", name),
                    });
                }
            },

            HirPatternKind::Or(patterns) => {
                if patterns.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "or-pattern must have at least one alternative".to_string(),
                    ));
                    return;
                }

                for sub_pattern in patterns.iter() {
                    self.check_pattern_against_type(sub_pattern, scrutinee_ty);
                }

                let mut first_bindings: Option<Vec<(StrId, HirType<'a, 'bump>)>> = None;
                for sub_pattern in patterns.iter() {
                    let mut bindings = Vec::new();
                    self.collect_pattern_bindings(sub_pattern, scrutinee_ty, &mut bindings);
                    bindings.sort_by(|(na, _), (nb, _)| na.cmp(nb));

                    match &first_bindings {
                        None => first_bindings = Some(bindings),
                        Some(expected) => {
                            if !self.bindings_match(expected, &bindings) {
                                let names = |v: &[(StrId, HirType<'a, 'bump>)]| {
                                    v.iter()
                                        .map(|(n, _)| n.to_string())
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                };
                                self.record(TypeErrorKind::Generic(format!(
                                    "all alternatives of an or-pattern must bind the same names with the same types: \
                                     found `{}` in one alternative but `{}` in another",
                                    names(expected),
                                    names(&bindings),
                                )));
                            }
                        }
                    }
                }
            }
        }
    }

    pub fn check_tail_expected(
        &mut self,
        stmt: &HirStmt<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        match stmt {
            HirStmt::Expr(e) => match expected {
                Some(exp) => self.check_expr_expected(e, exp),
                None => self.check_expr(e),
            },
            HirStmt::Match { expr, arms, span } => {
                self.set_span(*span);
                self.check_match_arms(expr, arms, expected)
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                self.set_span(*span);
                self.check_if_branches(cond, then_block, *else_block, expected)
                    .unwrap_or(HirType::synthetic(HirTypeKind::Void))
            }
            HirStmt::Block { body, span } => {
                self.set_span(*span);
                self.check_block_body(body, expected)
                    .unwrap_or(HirType::synthetic(HirTypeKind::Void))
            }
            other => self
                .check_stmt(other)
                .unwrap_or(HirType::synthetic(HirTypeKind::Void)),
        }
    }

    pub fn check_block_tail_expected(
        &mut self,
        body: &HirStmt<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        let HirStmt::Block { body, span: _ } = body else {
            unreachable!()
        };
        let Some((last, rest)) = body.split_last() else {
            return HirType::synthetic(HirTypeKind::Void);
        };
        for s in rest {
            self.check_stmt(s);
        }
        self.check_tail_expected(last, expected)
    }

    pub fn check_match_exhaustiveness(
        &mut self,
        scrutinee_ty: &HirType<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
    ) {
        if matches!(scrutinee_ty.kind, HirTypeKind::Unknown) {
            return;
        }

        let has_catch_all = arms.iter().any(|arm| {
            arm.guard.is_none()
                && matches!(
                    arm.pattern.kind,
                    HirPatternKind::Wildcard | HirPatternKind::Ident(_, _)
                )
        });
        if has_catch_all {
            return;
        }

        match &scrutinee_ty.kind {
            HirTypeKind::Nullable(_) => {
                let has_null = arms.iter().any(|arm| {
                    arm.guard.is_none() && matches!(arm.pattern.kind, HirPatternKind::Null)
                });
                if !has_null {
                    self.record(TypeErrorKind::Generic(format!(
                        "non-exhaustive match on `{}`: missing a `null` arm",
                        type_to_string(scrutinee_ty, &self.context.string_pool)
                    )));
                }
            }
            HirTypeKind::Enum {
                name: enum_name, ..
            } => {
                let Some(def) = self.context.get_enum(enum_name) else {
                    return;
                };
                let covered: std::collections::HashSet<StrId, FxHashBuilder> = arms
                    .iter()
                    .filter(|arm| arm.guard.is_none())
                    .filter_map(|arm| match arm.pattern.kind {
                        HirPatternKind::EnumVariant { variant, .. } => Some(variant),
                        HirPatternKind::Struct { name, .. } => Some(name),
                        _ => None,
                    })
                    .collect();
                let missing: Vec<&str> = def
                    .variants
                    .iter()
                    .filter(|v| !covered.contains(&v.name))
                    .map(|v| v.name.as_str())
                    .collect();
                if !missing.is_empty() {
                    self.record(TypeErrorKind::Generic(format!(
                        "non-exhaustive match on enum `{}`: missing variant(s) {}",
                        enum_name,
                        missing.join(", ")
                    )));
                }
            }

            HirTypeKind::Boolean => {
                let mut has_true = false;
                let mut has_false = false;
                for arm in arms.iter().filter(|a| a.guard.is_none()) {
                    match arm.pattern.kind {
                        HirPatternKind::Boolean(true) => has_true = true,
                        HirPatternKind::Boolean(false) => has_false = true,
                        _ => {}
                    }
                }
                if !(has_true && has_false) {
                    self.record(TypeErrorKind::Generic(
                        "non-exhaustive match on `bool`: requires a wildcard (`_`) arm or both `true` and `false` arms".to_string()
                    ));
                }
            }

            HirTypeKind::I8
            | HirTypeKind::I16
            | HirTypeKind::I32
            | HirTypeKind::I64
            | HirTypeKind::I128
            | HirTypeKind::U8
            | HirTypeKind::U16
            | HirTypeKind::U32
            | HirTypeKind::U64
            | HirTypeKind::U128
            | HirTypeKind::Usize
            | HirTypeKind::Isize
            | HirTypeKind::String
            | HirTypeKind::Char => {
                self.record(TypeErrorKind::Generic(format!(
                    "non-exhaustive match on `{}`: requires a wildcard (`_`) or binding (catch-all) arm",
                    type_to_string(scrutinee_ty, &self.context.string_pool)
                )));
            }

            _ => {} // structs/tuples/etc: not enforced yet
        }
    }

    pub fn join_value_types(&mut self, branches: &[HirType<'a, 'bump>]) -> HirType<'a, 'bump> {
        let nullable = |s: &Self, t: HirType<'a, 'bump>| match t.kind {
            HirTypeKind::Nullable(_) | HirTypeKind::Null => t,
            _ => HirType::new(HirTypeKind::Nullable(s.context.bump.alloc_value(t)), t.span),
        };
        let mut result: Option<HirType<'a, 'bump>> = None;
        for ty in branches {
            if matches!(ty.kind, HirTypeKind::Never) {
                continue;
            }
            result = Some(match result {
                None => *ty,
                Some(prev)
                    if matches!(prev.kind, HirTypeKind::Null)
                        && !matches!(ty.kind, HirTypeKind::Null) =>
                {
                    nullable(self, *ty)
                }
                Some(prev) if matches!(ty.kind, HirTypeKind::Null) => nullable(self, prev),
                Some(prev) => {
                    let c = self.types_compatible(&prev, ty);
                    self.recover(c, ());
                    prev
                }
            });
        }
        result.unwrap_or(HirType::synthetic(HirTypeKind::Never))
    }

    pub fn scrutinee_binding_mode(&self, scrutinee: &HirExpr<'a, 'bump>) -> BindingMode {
        match scrutinee {
            HirExpr::Ref { ref_kind, .. } => BindingMode::ByRef(*ref_kind),
            HirExpr::This { .. } => match self.local_ref_kind.get(&self.this_id) {
                Some(rk) => BindingMode::ByRef(*rk),
                None => BindingMode::ByValue,
            },

            HirExpr::Ident(name, _) => match self.local_ref_kind.get(name) {
                Some(rk) => BindingMode::ByRef(*rk),
                None => BindingMode::ByValue,
            },
            _ => BindingMode::ByValue,
        }
    }

    pub fn join_branch_types(
        &mut self,
        branches: &[HirType<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        if expected.is_some() {
            return self.join_value_types(branches);
        }

        let prev = self.suppress_errors;
        self.suppress_errors = true;
        let joined = self.join_value_types(branches);
        self.suppress_errors = prev;
        joined
    }

    pub fn peel_indirections(mut t: HirType<'a, 'bump>) -> HirType<'a, 'bump> {
        while let HirTypeKind::Ref { inner, .. } = t.kind {
            t = *inner;
        }
        t
    }
    pub fn peel_nullable<'x>(t: &'x HirType<'a, 'bump>) -> &'x HirType<'a, 'bump> {
        match &t.kind {
            HirTypeKind::Nullable(i) => i,
            _ => t,
        }
    }
}

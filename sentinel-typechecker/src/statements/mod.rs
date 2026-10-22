use ir::{
    borrow_checker::{Bound, LoanId},
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, HirStmt, HirType, HirTypeKind, StrId},
    span::SourceSpan,
};

use crate::{
    TypeChecker,
    initialization::{InitNode, InitStatus},
    naming::type_to_string,
    type_checker::Occurrence,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn check_package_stmt(
        &mut self,
        path: &ir::hir::Path<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        if self
            .context
            .dep_graph
            .borrow()
            .resolve_module_path(&path.path)
            .is_none()
        {
            let path_str = path
                .path
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            self.record(TypeErrorKind::Generic(format!(
                "cannot resolve package path `{}`",
                path_str
            )));
        }
        None
    }

    pub fn check_import_stmt(
        &mut self,
        path: &ir::hir::Path<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        if self
            .context
            .dep_graph
            .borrow()
            .resolve_module_path(&path.path)
            .is_none()
        {
            let path_str = path
                .path
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            self.record(TypeErrorKind::Generic(format!(
                "cannot resolve imported module `{}`",
                path_str
            )));
        }
        None
    }

    pub fn check_unsafe_stmt(&mut self, body: &&HirStmt<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.unsafe_depth += 1;
        let gotten = match **body {
            HirStmt::Block {
                body: ref stmts, ..
            } => self.check_block_with(stmts, true),
            _ => self.check_stmt(body),
        };
        self.unsafe_depth -= 1;
        gotten
    }

    pub fn check_const_stmt(
        &mut self,
        const_stmt: &&ir::hir::ConstStmt<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        let value_type = self.check_expr(&const_stmt.value);
        let result = self.types_compatible(&const_stmt.ty, &value_type);
        self.recover(result, ());
        if self.context.get_variable(const_stmt.name).is_some() {
            self.record(TypeErrorKind::VariableAlreadyExists {
                var_name: const_stmt.name.to_string(),
            });
        }
        let symbol_id = self.mint_symbol_id();
        self.context
            .add_variable(const_stmt.name, const_stmt.ty, symbol_id);
        None
    }

    pub fn check_break_stmt(
        &mut self,
        expr: &Option<&HirExpr<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) -> Option<HirType<'a, 'bump>> {
        if !self.context.in_loop {
            self.record(TypeErrorKind::BreakOutsideLoop);
        }
        if let Some(e) = expr {
            let expr_type = self.check_expr(e);
            self.check_and_record_value_use(e, &expr_type);
            if let Some(expected_return) = self.context.current_return_type {
                let result = self.types_compatible(&expected_return, &expr_type);
                self.recover(result, ());
            }
        }
        Some(HirType::never(span))
    }

    pub fn check_block_stmt(&mut self, body: &&[HirStmt<'a, 'bump>]) -> Option<HirType<'a, 'bump>> {
        self.check_block_with(body, false)
    }

    /// `promote_facts`: the block runs unconditionally (not a loop body or branch), so the
    /// arithmetic facts it assumed directly survive its end, minus any that mention its locals
    /// or a `mut` variable.
    pub fn check_block_with(
        &mut self,
        body: &&[HirStmt<'a, 'bump>],
        promote_facts: bool,
    ) -> Option<HirType<'a, 'bump>> {
        self.borrow_checker.begin_scope();
        let mut block_context = self.context.create_child_scope();

        let local_names: Vec<StrId> = body
            .iter()
            .filter_map(|s| match s {
                HirStmt::Let { name, .. } => Some(*name),
                _ => None,
            })
            .collect();

        let mut value = None;
        let mut tail_borrows: Vec<StrId> = Vec::new();
        for (i, stmt) in body.iter().enumerate() {
            // Arm tail tracking for the last statement only.
            let tail_key = match stmt {
                HirStmt::Expr(e) if i + 1 == body.len() => Some(Self::expr_key(e)),
                _ => None,
            };
            let prev_tail = tail_key.map(|k| {
                self.tail_borrows.clear();
                self.tail_expr_key.replace(k)
            });

            std::mem::swap(&mut self.context, &mut block_context);
            value = self.check_stmt(stmt);
            std::mem::swap(&mut self.context, &mut block_context);

            if let Some(prev) = prev_tail {
                self.tail_expr_key = prev;
                tail_borrows = std::mem::take(&mut self.tail_borrows);
            }

            let dead_loans: Vec<LoanId> = if self.loan_owners.is_empty() || local_names.is_empty() {
                Vec::new()
            } else {
                let after_point = self.stmt_after_points.get(&Self::stmt_key(stmt)).copied();
                self.loan_owners
                    .iter()
                    .filter(|(_, owner)| local_names.contains(owner))
                    .filter(|(_, owner)| match after_point {
                        Some(p) => !self.local_used_after(p, **owner),
                        None => !body[(i + 1)..]
                            .iter()
                            .any(|s| self.stmt_references_local(s, **owner)),
                    })
                    .map(|(&loan_id, _)| loan_id)
                    .collect()
            };
            for loan_id in dead_loans {
                self.borrow_checker.end_loan_now(loan_id);
                self.loan_owners.remove(&loan_id);
            }
        }

        // Locals whose drop is observable (guard-like). Restricting to these avoids
        // false positives on plain reference-typed locals.
        let dropped: Vec<(StrId, StrId)> = local_names
            .iter()
            .rev()
            .filter_map(|n| {
                let (_, ty) = block_context.get_variable(n)?;
                match ty.kind {
                    HirTypeKind::Struct { name, .. } | HirTypeKind::Enum { name, .. }
                        if self.copy_analysis.borrow().implements_drop(name) =>
                    {
                        Some((*n, name))
                    }
                    _ => None,
                }
            })
            .collect();

        // The tail value leaves the block with no owner, so no later use can catch it.
        let mut reported: Vec<StrId> = Vec::new();
        for l in &tail_borrows {
            if reported.contains(l) {
                continue;
            }
            if let Some((_, owner)) = dropped.iter().find(|(n, _)| n == l) {
                reported.push(*l);
                self.record(TypeErrorKind::Generic(format!(
                    "this block's value borrows from `{}` (a `{}`), which is dropped at the \
                     end of the block; the reference cannot outlive it",
                    *l, *owner,
                )));
            }
        }

        // Anything derived from a dropped local is now invalid. Later uses through an
        // outer local's provenance fail in `check_use` (`InvalidatedReference`).
        for (n, _) in &dropped {
            self.borrow_checker.kill_local_storage(*n);
        }

        // Falling off the end of a function body is an exit like `return`, and the facts that
        // hold on that path are exactly the ones in this scope. (Not gated on `Never` alone:
        // a diverging tail has already been checked at its `return`.)
        if self.fn_body_ptr == Some(body.as_ptr() as usize)
            && !body.is_empty()
            && !matches!(&value, Some(t) if matches!(t.kind, HirTypeKind::Never))
        {
            if let Some(f) = self.current_func {
                self.check_ensures_at_fallthrough(&f);
            }
        }

        let promoted: Vec<(Bound, Bound, bool)> = if promote_facts {
            self.borrow_checker
                .current_scope_facts()
                .into_iter()
                .filter(|(l, r, _)| {
                    !Self::bound_mentions(l, &local_names)
                        && !Self::bound_mentions(r, &local_names)
                        && self.fact_is_trackable(l, r)
                })
                .collect()
        } else {
            Vec::new()
        };
        self.borrow_checker.end_scope();
        for (l, r, eq) in promoted {
            if eq {
                self.borrow_checker.assume_equal_scoped(l, r);
            } else {
                self.borrow_checker.assume_not_equal_scoped(l, r);
            }
        }
        // sweep anything that reached scope-end without an early kill
        if !self.loan_owners.is_empty() {
            self.loan_owners
                .retain(|_, owner| !local_names.contains(owner));
        }
        for name in &local_names {
            self.local_provenance_place.remove(name);
            self.local_provenance.remove(name);
            self.local_ref_kind.remove(name);
        }
        value
    }

    pub fn check_for_stmt(
        &mut self,
        init: &Option<&HirStmt<'a, 'bump>>,
        condition: &Option<&HirExpr<'a, 'bump>>,
        increment: &Option<&HirExpr<'a, 'bump>>,
        body: &&HirStmt<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        if let Some(init_stmt) = init {
            self.check_stmt(init_stmt);
        }

        if let Some(cond) = condition {
            self.check_if_condition(*cond);
        }

        self.kill_facts_about_mutables();
        self.context.enter_loop();

        let entry_state = self.move_state.clone();
        let init_entry = self.init_state.clone();
        let (converged_entry, converged_init) =
            self.converge_loop_states(body, entry_state, init_entry);
        self.move_state = converged_entry;
        self.init_state = converged_init;

        self.enter_loop_body(body);
        self.check_stmt(body);

        self.context.exit_loop();

        if let Some(inc) = increment {
            let snap = self.snapshot_temp_loans();
            self.check_expr(inc);
            self.end_temp_call_loans(snap);
            self.end_temp_closure_loans(snap);
        }

        None
    }

    pub fn check_while_stmt(
        &mut self,
        cond: &&HirExpr<'a, 'bump>,
        body: &&HirStmt<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        self.check_if_condition(*cond);

        self.kill_facts_about_mutables();
        self.context.enter_loop();

        let entry_state = self.move_state.clone();
        let init_entry = self.init_state.clone();
        let (converged_entry, converged_init) =
            self.converge_loop_states(body, entry_state, init_entry);
        self.move_state = converged_entry;
        self.init_state = converged_init;

        self.enter_loop_body(body);
        self.check_stmt(body);

        self.context.exit_loop();
        None
    }

    pub fn check_expr_stmt(&mut self, e: &&HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        let snap = self.snapshot_temp_loans();
        let ty = self.check_expr(e);

        // `call(..);` as a statement always runs, so what the callee's `ensures` promised about
        // its arguments holds for the rest of this scope.
        self.apply_unconditional_call_facts(e);

        if self.tail_expr_key == Some(Self::expr_key(e)) && self.return_type_may_alias(&ty) {
            let found: Vec<StrId> = self.call_loan_order[snap.call..]
                .iter()
                .filter_map(|&(key, id)| (self.call_loans.get(&key) == Some(&id)).then_some(id))
                .filter_map(|id| self.borrow_checker.loan(id))
                .filter_map(|loan| self.borrow_checker.root_local_of(loan.place))
                .collect();

            self.tail_borrows = found;
        }

        self.end_temp_call_loans(snap);
        self.end_temp_closure_loans(snap);
        Some(ty)
    }

    pub fn check_return_stmt(
        &mut self,
        expr: &Option<&HirExpr<'a, 'bump>>,
        span: SourceSpan<'a>,
    ) -> Option<HirType<'a, 'bump>> {
        // Every exit must establish the function's relational `ensures` conditions, with or
        // without a returned value. (Falling off the end of a `void` function is the other exit:
        // `check_ensures_at_fallthrough`.)
        if let Some(f) = self.current_func {
            self.check_ensures_conditions_at_exit(&f);
        }

        if let Some(e) = expr {
            let skip_read = match self.current_func {
                Some(f) => self.check_return_against_ensures(&f, Some(*e)),
                None => false,
            };
            let prev_suppress = self.suppress_init_read;
            if skip_read {
                self.suppress_init_read = true;
            }

            let expected_return = self.context.current_return_type;
            let expr_type = match expected_return {
                Some(ret) => self.check_expr_expected(e, &ret),
                None => self.check_expr(e),
            };
            self.check_and_record_value_use(e, &expr_type);
            let dangling = self.check_no_dangling_pointer(e);
            self.recover(dangling, ());
            if let Some(expected_return) = expected_return {
                self.recover(self.types_compatible(&expected_return, &expr_type), ());
            }

            self.suppress_init_read = prev_suppress;
        } else if let Some(expected_return) = self.context.current_return_type {
            if expected_return.kind != HirTypeKind::Void {
                self.record(TypeErrorKind::InvalidReturnType {
                    expected: type_to_string(&expected_return, &self.context.string_pool)
                        .to_string(),
                    found: "void".to_string(),
                });
            }
        }
        Some(HirType::never(span))
    }

    pub fn init_struct_with_uninit_fields(
        &mut self,
        root: StrId,
        ty: &HirType<'a, 'bump>,
        uninit: &[StrId],
    ) {
        let HirTypeKind::Struct { name, .. } = ty.kind else {
            return;
        };
        let Some(def) = self.context.get_struct(name.as_str()) else {
            return;
        };
        let map = def
            .fields
            .iter()
            .map(|f| {
                let s = if uninit.contains(&f.name) {
                    InitStatus::Uninitialized
                } else {
                    InitStatus::Initialized
                };
                (f.name, InitNode::Whole(s))
            })
            .collect();
        self.set_init_node(root, InitNode::Struct(map));
    }

    pub fn check_let_stmt(
        &mut self,
        name: &StrId,
        ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
        mutable: &bool,
        span: &SourceSpan<'a>,
        inferred_ty: bool,
    ) -> Option<HirType<'a, 'bump>> {
        let is_wildcard = name == "_";

        if !inferred_ty {
            self.record_type_occurrences(ty);
        }

        if let Some((r, p)) = self.static_field_path(value) {
            if self.is_non_null(r, &p) {
                self.mark_non_null(*name, &[]);
            }
        }

        if !is_wildcard && self.context.variables.contains_key(name) {
            self.record(TypeErrorKind::VariableAlreadyExists {
                var_name: name.to_string(),
            });
        }

        let temp_snap = self.snapshot_temp_loans();
        let value_type = self.check_expr_expected(value, ty);
        let ty = if matches!(ty.kind, HirTypeKind::Unknown) {
            &value_type
        } else {
            ty
        };

        // `let x = call(..);` always runs the call: assume the callee's relational `ensures`.
        self.apply_unconditional_call_facts(value);
        // `name` is a fresh variable: facts that mention the name (about an earlier variable it
        // shadows, including ones the line above just applied) are not about it.
        self.kill_facts_about(*name);

        let is_uninit_value = matches!(value, HirExpr::Uninit { .. });
        if !is_uninit_value {
            self.check_and_record_value_use(value, &value_type);
        }

        let result = self.types_compatible(ty, &value_type);
        self.recover(result, ());

        if self.expr_is_dangling(value) {
            self.context.mark_dangling(name.clone());
        }

        let symbol_id = self.mint_symbol_id();

        self.context
            .add_variable_with_mutability(name, *ty, *mutable, symbol_id);
        self.borrow_checker.declare_local(*name);
        self.bind_lambda_ensures(*name, value, *mutable);
        let local_id = self.local_table.register(*name);
        self.move_state.clear(local_id);
        self.occurrences.push(Occurrence {
            span: *span,
            name: *name,
            hir_name: *name,
            ty: *ty,
            module_idx: self.context.current_module_idx,
            symbol: symbol_id,
            is_decl: true,
        });

        if is_uninit_value {
            self.mark_whole_uninit(*name);
        } else {
            self.mark_whole_init(*name);
            if let HirExpr::StructInit { args, .. } = value {
                let uninit: Vec<StrId> = args
                    .iter()
                    .filter(|fi| matches!(fi.value, HirExpr::Uninit { .. }))
                    .map(|fi| fi.name)
                    .collect();
                if !uninit.is_empty() {
                    self.init_struct_with_uninit_fields(*name, ty, &uninit);
                }
            }
        }

        if matches!(
            ty.kind,
            HirTypeKind::SafePointer { .. } | HirTypeKind::UnsafePointer { .. }
        ) {
            if let Some(place) = self.resolve_place(value) {
                if let Some(&(base, ref offset)) = self.borrow_checker.pointee_of(place) {
                    let declared = *self.borrow_checker.local_place(*name).unwrap();
                    self.borrow_checker
                        .record_pointee(declared, base, offset.clone());
                }
            }
        }

        if let Some(loan_ids) = self.closure_loans.remove(&Self::expr_key(value)) {
            for loan_id in loan_ids {
                self.loan_owners.insert(loan_id, *name);
            }
        }

        if let Some(loan_id) = self.call_loans.remove(&Self::expr_key(value)) {
            self.loan_owners.insert(loan_id, *name);
            if let Some(loan) = self.borrow_checker.loan(loan_id) {
                self.local_provenance_place.insert(*name, loan.place);
                self.local_provenance.insert(*name, loan.provenance_id);
            }
        } else if let HirExpr::Ref {
            expr: ref_target, ..
        } = value
        {
            if let Some(place) = self.resolve_place(ref_target) {
                self.local_provenance_place.insert(*name, place);
                if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                    self.loan_owners.insert(loan_id, *name);
                    if let Some(loan) = self.borrow_checker.loan(loan_id) {
                        self.local_provenance.insert(*name, loan.provenance_id);
                    }
                }
            }
        }

        let leftover = &self.call_loan_order[temp_snap.call..];

        if self.return_type_may_alias(ty) {
            for (key, id) in leftover {
                if self.call_loans.get(&key) != Some(&id) {
                    continue;
                }

                self.call_loans.remove(&key);
                self.loan_owners.insert(*id, *name);

                if let Some(loan) = self.borrow_checker.loan(*id) {
                    self.local_provenance_place.insert(*name, loan.place);
                    self.local_provenance.insert(*name, loan.provenance_id);
                }
            }
        } else {
            self.end_temp_call_loans(temp_snap);
        }

        None
    }
}

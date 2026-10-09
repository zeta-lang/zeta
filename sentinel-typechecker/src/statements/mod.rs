use ir::{
    borrow_checker::LoanId,
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, HirStmt, HirType, StrId},
    span::SourceSpan,
};

use crate::{
    TypeChecker,
    initialization::{InitNode, InitStatus},
    naming::type_to_string,
    str_id_to_string,
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
        let gotten = self.check_stmt(body);
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
        let var_name = str_id_to_string(const_stmt.name);
        if self.context.get_variable(&var_name).is_some() {
            self.record(TypeErrorKind::VariableAlreadyExists {
                var_name: var_name.clone(),
            });
        }
        let symbol_id = self.mint_symbol_id();
        self.context
            .add_variable(var_name, const_stmt.ty, symbol_id);
        None
    }

    pub fn check_break_stmt(
        &mut self,
        expr: &Option<&HirExpr<'a, 'bump>>,
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
        Some(HirType::Never)
    }

    pub fn check_block_stmt(&mut self, body: &&[HirStmt<'a, 'bump>]) -> Option<HirType<'a, 'bump>> {
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

            let old_context = std::mem::replace(&mut self.context, block_context);
            value = self.check_stmt(stmt);
            block_context = self.context.clone();
            self.context = old_context;

            if let Some(prev) = prev_tail {
                self.tail_expr_key = prev;
                tail_borrows = std::mem::take(&mut self.tail_borrows);
            }

            let after_point = self.stmt_after_points.get(&Self::stmt_key(stmt)).copied();

            let dead_loans: Vec<LoanId> = self
                .loan_owners
                .iter()
                .filter(|(_, owner)| local_names.contains(owner))
                .filter(|(_, owner)| match after_point {
                    Some(p) => !self.local_used_after(p, **owner),
                    None => !body[(i + 1)..]
                        .iter()
                        .any(|s| self.stmt_references_local(s, **owner)),
                })
                .map(|(&loan_id, _)| loan_id)
                .collect();
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
                let (_, ty) = block_context.get_variable(&str_id_to_string(*n))?;
                match ty {
                    HirType::Struct { name, .. } | HirType::Enum { name, .. }
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
                    str_id_to_string(*l),
                    str_id_to_string(*owner),
                )));
            }
        }

        // Anything derived from a dropped local is now invalid. Later uses through an
        // outer local's provenance fail in `check_use` (`InvalidatedReference`).
        for (n, _) in &dropped {
            self.borrow_checker.kill_local_storage(*n);
        }

        self.borrow_checker.end_scope();
        // sweep anything that reached scope-end without an early kill
        self.loan_owners
            .retain(|_, owner| !local_names.contains(owner));
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
            let cond_type = self.check_expr(cond);
            if cond_type != HirType::Boolean {
                self.record(TypeErrorKind::TypeMismatch {
                    expected: "bool".to_string(),
                    found: type_to_string(&cond_type),
                });
            }
        }

        self.context.enter_loop();

        let entry_state = self.move_state.clone();
        let init_entry = self.init_state.clone();
        let (converged_entry, converged_init) =
            self.converge_loop_states(body, entry_state, init_entry);
        self.move_state = converged_entry;
        self.init_state = converged_init;

        self.check_stmt(body);

        self.context.exit_loop();

        if let Some(inc) = increment {
            self.check_expr(inc);
        }

        None
    }

    pub fn check_while_stmt(
        &mut self,
        cond: &&HirExpr<'a, 'bump>,
        body: &&HirStmt<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        let cond_type = self.check_expr(cond);
        if cond_type != HirType::Boolean {
            self.record(TypeErrorKind::TypeMismatch {
                expected: "bool".to_string(),
                found: type_to_string(&cond_type),
            });
        }

        self.context.enter_loop();

        let entry_state = self.move_state.clone();
        let init_entry = self.init_state.clone();
        let (converged_entry, converged_init) =
            self.converge_loop_states(body, entry_state, init_entry);
        self.move_state = converged_entry;
        self.init_state = converged_init;

        self.check_stmt(body);

        self.context.exit_loop();
        None
    }

    pub fn check_expr_stmt(&mut self, e: &&HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        let snap = self.snapshot_call_loan_keys();
        let ty = self.check_expr(e);

        if self.tail_expr_key == Some(Self::expr_key(e)) && self.return_type_may_alias(&ty) {
            let found: Vec<StrId> = self
                .call_loans
                .iter()
                .filter(|(k, _)| !snap.contains(*k))
                .filter_map(|(_, &id)| self.borrow_checker.loan(id))
                .filter_map(|l| self.borrow_checker.root_local_of(l.place))
                .collect();
            self.tail_borrows = found;
        }

        self.end_temp_call_loans(&snap);
        self.end_temp_closure_loans(&snap);
        Some(ty)
    }

    pub fn check_return_stmt(
        &mut self,
        expr: &Option<&HirExpr<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        if let Some(e) = expr {
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
        } else if let Some(expected_return) = self.context.current_return_type {
            if expected_return != HirType::Void {
                self.record(TypeErrorKind::InvalidReturnType {
                    expected: type_to_string(&expected_return),
                    found: "void".to_string(),
                });
            }
        }
        Some(HirType::Never)
    }

    pub fn init_struct_with_uninit_fields(
        &mut self,
        root: StrId,
        ty: &HirType<'a, 'bump>,
        uninit: &[StrId],
    ) {
        let HirType::Struct { name, .. } = ty else {
            return;
        };
        let Some(def) = self.context.get_struct(&str_id_to_string(*name)) else {
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
        self.init_state.insert(root, InitNode::Struct(map));
    }

    pub fn check_let_stmt(
        &mut self,
        name: &StrId,
        ty: &HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
        mutable: &bool,
        span: &SourceSpan<'a>,
    ) -> Option<HirType<'a, 'bump>> {
        let var_name = str_id_to_string(*name);
        let is_wildcard = var_name == "_";

        if let Some((r, p)) = self.static_field_path(value) {
            if self.is_non_null(r, &p) {
                self.mark_non_null(*name, &[]);
            }
        }

        if !is_wildcard && self.context.variables.contains_key(&var_name) {
            self.record(TypeErrorKind::VariableAlreadyExists {
                var_name: var_name.clone(),
            });
        }

        let temp_snap = self.snapshot_call_loan_keys();
        let value_type = self.check_expr_expected(value, ty);

        let is_uninit_value = matches!(value, HirExpr::Uninit { .. });
        if !is_uninit_value {
            self.check_and_record_value_use(value, &value_type);
        }

        let result = self.types_compatible(ty, &value_type);
        self.recover(result, ());

        if self.expr_is_dangling(value) {
            self.context.mark_dangling(var_name.clone());
        }

        let symbol_id = self.mint_symbol_id();
        self.context
            .add_variable_with_mutability(var_name, *ty, *mutable, symbol_id);
        self.borrow_checker.declare_local(*name);
        self.move_state.clear(*name);
        self.occurrences.push((
            *span,
            *name,
            *ty,
            self.context.current_module_idx,
            symbol_id,
            true,
        ));

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
            ty,
            HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
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

        let leftover: Vec<usize> = self
            .call_loans
            .keys()
            .copied()
            .filter(|k| !temp_snap.contains(k))
            .collect();
        if self.return_type_may_alias(ty) {
            // the value may still point through these temporaries: the new local keeps them alive
            for k in leftover {
                if let Some(id) = self.call_loans.remove(&k) {
                    self.loan_owners.insert(id, *name);
                }
            }
        } else {
            // copied-out value (`i64` etc.): nothing borrows from the source any more
            self.end_temp_call_loans(&temp_snap);
        }

        None
    }
}

use ir::{
    borrow_checker::{LoanId, PlaceId},
    errors::type_error::TypeErrorKind,
    hir::{
        HirExpr, HirMatchArm, HirPattern, HirStmt, HirType, ProvenanceAnnotation, RefKind, StrId,
    },
    ir_hasher::FxHashBuilder,
};

use crate::{
    initialization::BindingMode, move_state::MoveState, naming::type_to_string, str_id_to_string,
    type_checker::NonNullState, TypeChecker,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn check_match_arms(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        let scrutinee_ty = self.check_expr(expr);

        self.check_match_exhaustiveness(&scrutinee_ty, arms);

        let move_state_before = self.move_state.clone();
        let mut arm_types = Vec::with_capacity(arms.len());
        let mut arm_move_states = Vec::with_capacity(arms.len());

        for arm in arms {
            let (arm_type, arm_move_state) =
                self.check_match_arm(expr, arm, &scrutinee_ty, expected, &move_state_before);

            arm_types.push(arm_type);
            arm_move_states.push(arm_move_state);
        }

        self.move_state = arm_move_states
            .into_iter()
            .fold(move_state_before, |acc, state| {
                MoveState::join(&acc, &state)
            });

        self.join_value_types(&arm_types)
    }

    pub fn check_match_arm(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        arm: &HirMatchArm<'a, 'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
        move_state_before: &MoveState,
    ) -> (HirType<'a, 'bump>, MoveState) {
        let binding_mode = self.scrutinee_binding_mode(scrutinee);
        let scrutinee_place = self.resolve_place(scrutinee);
        let scrutinee_provenance = self.infer_provenance(scrutinee);

        self.binding_mode_backfill
            .insert(Self::expr_key(scrutinee), binding_mode);

        self.move_state = move_state_before.clone();

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

        self.check_match_arm_guard(arm.guard);

        let arm_type = self.check_block_tail_expected(arm.body, expected);

        self.context = old_context;
        self.borrow_checker.end_scope();

        self.cleanup_match_arm_bindings(&arm.pattern, scrutinee_ty);

        (arm_type, self.move_state.clone())
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

        if guard_type != HirType::Boolean {
            self.record(TypeErrorKind::TypeMismatch {
                expected: "bool".to_string(),
                found: type_to_string(&guard_type),
            });
        }
    }

    pub fn cleanup_match_arm_bindings(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
    ) {
        let mut bound = Vec::new();
        self.collect_pattern_bindings(pattern, scrutinee_ty, &mut bound);

        for (name, _) in bound {
            self.local_provenance_place.remove(&name);
            self.local_ref_kind.remove(&name);
        }
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
            let old_context = std::mem::replace(&mut self.context, block_context);

            value = if Some(i) == last_idx {
                Some(self.check_tail_expected(stmt, expected))
            } else {
                self.check_stmt(stmt)
            };

            block_context = self.context.clone();
            self.context = old_context;

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
        let after_point = self
            .stmt_after_points
            .get(&Self::stmt_key(&body[statement_index]))
            .copied();

        let dead_loans: Vec<LoanId> = self
            .loan_owners
            .iter()
            .filter(|(_, owner)| local_names.contains(owner))
            .filter(|(_, &owner)| match after_point {
                Some(point) => !self.local_used_after(point, owner),
                None => !body[(statement_index + 1)..]
                    .iter()
                    .any(|stmt| self.stmt_references_local(stmt, owner)),
            })
            .map(|(&loan_id, _)| loan_id)
            .collect();

        for loan_id in dead_loans {
            self.borrow_checker.end_loan_now(loan_id);
            self.loan_owners.remove(&loan_id);
        }
    }

    pub fn cleanup_block_locals(&mut self, local_names: &[StrId]) {
        self.loan_owners
            .retain(|_, owner| !local_names.contains(owner));

        for name in local_names {
            self.local_provenance_place.remove(name);
            self.local_ref_kind.remove(name);
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

        let move_state_before = self.move_state.clone();
        let non_null_before = self.non_null_state.clone();

        let then_value = self.check_if_then_branch(cond, then_block, expected);
        let then_move_state = self.move_state.clone();
        let then_non_null = self.non_null_state.clone();
        let then_diverges = matches!(then_value, HirType::Never);

        self.restore_state_for_else_branch(cond, &move_state_before, &non_null_before);

        let else_value = self.check_if_else_branch(cond, else_block, expected);
        let else_move_state = self.move_state.clone();
        let else_non_null = self.non_null_state.clone();

        self.join_if_branch_states(
            then_move_state,
            else_move_state,
            then_non_null,
            else_non_null,
            then_diverges,
            else_value
                .as_ref()
                .is_some_and(|ty| matches!(ty, HirType::Never)),
        );

        else_value.map(|else_type| self.join_value_types(&[then_value, else_type]))
    }

    pub fn check_if_condition(&mut self, cond: &HirExpr<'a, 'bump>) {
        let snap = self.snapshot_call_loan_keys();
        let cond_type = self.check_expr(cond);
        self.end_temp_call_loans(&snap);

        if cond_type != HirType::Boolean {
            self.record(TypeErrorKind::TypeMismatch {
                expected: "bool".to_string(),
                found: type_to_string(&cond_type),
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
        let mut then_value = HirType::Void;

        if let Some((last, rest)) = then_block.split_last() {
            for stmt in rest {
                let old_context = std::mem::replace(&mut self.context, then_context);

                self.check_stmt(stmt);

                then_context = self.context.clone();
                self.context = old_context;
            }

            let old_context = std::mem::replace(&mut self.context, then_context);

            then_value = self.check_tail_expected(last, expected);

            self.context = old_context;
        }

        self.borrow_checker.end_scope();

        then_value
    }

    pub fn assume_if_condition_is_true(&mut self, cond: &HirExpr<'a, 'bump>) {
        let fact = self.condition_to_fact(cond);
        let non_null_fact = self.condition_to_non_null_fact(cond);
        let place_fact = self.condition_to_place_fact(cond);

        if let Some((lhs, rhs, is_equal)) = &fact {
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
        let fact = self.condition_to_fact(cond);
        let non_null_fact = self.condition_to_non_null_fact(cond);
        let place_fact = self.condition_to_place_fact(cond);

        if let Some((lhs, rhs, is_equal)) = &fact {
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

    pub fn restore_state_for_else_branch(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        move_state_before: &MoveState,
        non_null_before: &NonNullState,
    ) {
        self.move_state = move_state_before.clone();
        self.non_null_state = non_null_before.clone();

        self.assume_if_condition_is_false(cond);
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
        self.move_state = MoveState::join(&then_move_state, &else_move_state);

        self.non_null_state = match (then_diverges, else_diverges) {
            (true, false) => else_non_null,
            (false, true) => then_non_null,
            _ => Self::join_non_null_states(&then_non_null, &else_non_null),
        };
    }

    pub fn register_pattern_bindings(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        mode: BindingMode,
        scrutinee_provenance: Option<ProvenanceAnnotation<'bump>>,
        scrutinee_place: Option<PlaceId>,
    ) {
        if let HirType::Nullable(inner) = scrutinee_ty {
            if !matches!(pattern, HirPattern::Null) {
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

        match pattern {
            HirPattern::Ident(name) => {
                let var_name = str_id_to_string(*name);
                let symbol_id = self.mint_symbol_id();
                let bound_ty = self.bind_leaf_type(*scrutinee_ty, mode, scrutinee_provenance);
                if self.context.get_variable(&var_name).is_some() {
                    self.record(TypeErrorKind::VariableAlreadyExists {
                        var_name: var_name.clone(),
                    });
                }
                self.context.add_variable(var_name, bound_ty, symbol_id);

                self.borrow_checker.declare_local(*name);

                match mode {
                    BindingMode::ByRef(rk) => {
                        self.local_ref_kind.insert(*name, rk);
                        if let Some(src_place) = scrutinee_place {
                            self.local_provenance_place.insert(*name, src_place);
                            let result = match rk {
                                RefKind::Unique => self.borrow_checker.borrow_mut(src_place),
                                RefKind::Alias => self.borrow_checker.borrow_alias(src_place),
                                RefKind::Shared => self.borrow_checker.borrow_shared(src_place),
                            };
                            match result {
                                Ok(loan_id) => {
                                    self.loan_owners.insert(loan_id, *name);
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

            HirPattern::EnumVariant {
                variant, bindings, ..
            } => {
                let HirType::Enum {
                    name: enum_name, ..
                } = scrutinee_ty
                else {
                    return;
                };
                let enum_name_str = str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == *variant) else {
                    return;
                };
                for (binding_name, field) in bindings.iter().zip(variant_def.fields.iter()) {
                    let var_name = str_id_to_string(*binding_name);
                    let symbol_id = self.mint_symbol_id();
                    let field_provenance =
                        Self::extend_provenance(scrutinee_provenance, field.name, &self.context);
                    let bound_ty = self.bind_leaf_type(field.field_type, mode, field_provenance);
                    if self.context.get_variable(&var_name).is_some() {
                        self.record(TypeErrorKind::VariableAlreadyExists {
                            var_name: var_name.clone(),
                        });
                    }
                    self.context.add_variable(var_name, bound_ty, symbol_id);
                }
            }

            HirPattern::Tuple(patterns) => {
                if let HirType::Tuple(elems) = scrutinee_ty {
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

            HirPattern::Array(patterns) => {
                let elem_ty = match scrutinee_ty {
                    HirType::Array(inner, _) | HirType::Slice(inner) => Some(**inner),
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

            HirPattern::Struct { name, fields } => match scrutinee_ty {
                HirType::Enum {
                    name: enum_name, ..
                } => {
                    let enum_name_str = str_id_to_string(*enum_name);
                    let Some(def) = self.context.get_enum(&enum_name_str) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == *name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_def) =
                            variant_def.fields.iter().find(|f| f.name == *field_name)
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
                            &field_def.field_type,
                            mode,
                            field_provenance,
                            scrutinee_place,
                        );
                    }
                }
                HirType::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let struct_name_str = str_id_to_string(*struct_name);
                    let Some(def) = self.context.get_struct(&struct_name_str) else {
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

            HirPattern::Or(patterns) => {
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

    pub fn bind_leaf_type(
        &self,
        field_ty: HirType<'a, 'bump>,
        mode: BindingMode,
        provenance: Option<ProvenanceAnnotation<'bump>>,
    ) -> HirType<'a, 'bump> {
        match mode {
            BindingMode::ByValue => field_ty,
            BindingMode::ByRef(rk) => HirType::Ref {
                inner: self.context.bump.alloc_value(field_ty),
                ref_kind: rk,
                provenance,
            },
        }
    }

    pub fn collect_pattern_bindings(
        &self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        out: &mut Vec<(StrId, HirType<'a, 'bump>)>,
    ) {
        if let HirType::Nullable(inner) = scrutinee_ty {
            if !matches!(pattern, HirPattern::Null) {
                let inner_ty = **inner;
                return self.collect_pattern_bindings(pattern, &inner_ty, out);
            }
        }

        match pattern {
            HirPattern::Ident(name) => {
                out.push((*name, *scrutinee_ty));
            }

            HirPattern::EnumVariant {
                variant, bindings, ..
            } => {
                let HirType::Enum {
                    name: enum_name, ..
                } = scrutinee_ty
                else {
                    return;
                };
                let enum_name_str = str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == *variant) else {
                    return;
                };
                for (binding_name, field) in bindings.iter().zip(variant_def.fields.iter()) {
                    out.push((*binding_name, field.field_type));
                }
            }

            HirPattern::Tuple(patterns) => {
                if let HirType::Tuple(elems) = scrutinee_ty {
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.collect_pattern_bindings(sub_pattern, elem_ty, out);
                    }
                }
            }

            HirPattern::Array(patterns) => {
                let elem_ty = match scrutinee_ty {
                    HirType::Array(inner, _) | HirType::Slice(inner) => Some(**inner),
                    _ => None,
                };
                if let Some(elem_ty) = elem_ty {
                    for sub_pattern in patterns.iter() {
                        self.collect_pattern_bindings(sub_pattern, &elem_ty, out);
                    }
                }
            }

            HirPattern::Struct { name, fields } => match scrutinee_ty {
                HirType::Enum {
                    name: enum_name, ..
                } => {
                    let enum_name_str = str_id_to_string(*enum_name);
                    let Some(def) = self.context.get_enum(&enum_name_str) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == *name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_def) =
                            variant_def.fields.iter().find(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        self.collect_pattern_bindings(sub_pattern, &field_def.field_type, out);
                    }
                }
                HirType::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let struct_name_str = str_id_to_string(*struct_name);
                    let Some(def) = self.context.get_struct(&struct_name_str) else {
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

            HirPattern::Or(patterns) => {
                if let Some(first) = patterns.first() {
                    self.collect_pattern_bindings(first, scrutinee_ty, out);
                }
            }

            HirPattern::Wildcard
            | HirPattern::Number(_)
            | HirPattern::String(_)
            | HirPattern::Boolean(_)
            | HirPattern::Null => {}
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

    pub fn check_pattern_against_type(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
    ) {
        if let HirType::Nullable(inner) = scrutinee_ty {
            if !matches!(pattern, HirPattern::Null) {
                let inner_ty = **inner;
                return self.check_pattern_against_type(pattern, &inner_ty);
            }
        }
        match pattern {
            HirPattern::Null => {
                if !matches!(scrutinee_ty, HirType::Nullable(_)) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty),
                        found: "null".to_string(),
                    });
                }
            }
            HirPattern::EnumVariant {
                variant, bindings, ..
            } => {
                let HirType::Enum {
                    name: enum_name, ..
                } = scrutinee_ty
                else {
                    return;
                };
                let enum_name_str = str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == *variant) else {
                    self.record(TypeErrorKind::Generic(format!(
                        "enum `{}` has no variant `{}`",
                        enum_name_str, variant
                    )));
                    return;
                };
                if bindings.len() != variant_def.fields.len() {
                    self.record(TypeErrorKind::Generic(format!(
                        "variant `{}::{}` has {} field(s), but the pattern binds {}",
                        enum_name_str,
                        variant,
                        variant_def.fields.len(),
                        bindings.len()
                    )));
                }
            }
            HirPattern::Boolean(_) => {
                if !matches!(scrutinee_ty, HirType::Boolean) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty),
                        found: "bool".to_string(),
                    });
                }
            }
            HirPattern::Number(_) => {
                if !self.is_integer(scrutinee_ty) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty),
                        found: "integer".to_string(),
                    });
                }
            }
            HirPattern::String(_) => {
                if !matches!(scrutinee_ty, HirType::String) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty),
                        found: "str".to_string(),
                    });
                }
            }

            HirPattern::Ident(_) | HirPattern::Wildcard => {}

            HirPattern::Tuple(patterns) => match scrutinee_ty {
                HirType::Tuple(elems) => {
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
                        expected: type_to_string(scrutinee_ty),
                        found: format!("tuple pattern with {} element(s)", patterns.len()),
                    });
                }
            },

            HirPattern::Array(patterns) => match scrutinee_ty {
                HirType::Array(inner, len) => {
                    if patterns.len() != *len {
                        self.record(TypeErrorKind::Generic(format!(
                            "array pattern has {} element(s), but the array type `{}` has length {}",
                            patterns.len(),
                            type_to_string(scrutinee_ty),
                            len
                        )));
                    }
                    for sub_pattern in patterns.iter() {
                        self.check_pattern_against_type(sub_pattern, inner);
                    }
                }
                HirType::Slice(inner) => {
                    for sub_pattern in patterns.iter() {
                        self.check_pattern_against_type(sub_pattern, inner);
                    }
                }
                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: type_to_string(scrutinee_ty),
                        found: format!("array pattern with {} element(s)", patterns.len()),
                    });
                }
            },

            HirPattern::Struct { name, fields } => match scrutinee_ty {
                HirType::Enum {
                    name: enum_name, ..
                } => {
                    let enum_name_str = str_id_to_string(*enum_name);
                    let Some(def) = self.context.get_enum(&enum_name_str) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == *name) else {
                        self.record(TypeErrorKind::Generic(format!(
                            "enum `{}` has no variant `{}`",
                            enum_name_str, name
                        )));
                        return;
                    };

                    let mut seen: std::collections::HashSet<StrId> =
                        std::collections::HashSet::new();
                    for (field_name, sub_pattern) in fields.iter() {
                        if !seen.insert(*field_name) {
                            self.record(TypeErrorKind::Generic(format!(
                                "field `{}` matched more than once in this pattern",
                                str_id_to_string(*field_name)
                            )));
                            continue;
                        }
                        let Some(field_def) =
                            variant_def.fields.iter().find(|f| f.name == *field_name)
                        else {
                            self.record(TypeErrorKind::Generic(format!(
                                "variant `{}::{}` has no field `{}`",
                                enum_name_str,
                                name,
                                str_id_to_string(*field_name)
                            )));
                            continue;
                        };
                        self.check_pattern_against_type(sub_pattern, &field_def.field_type);
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
                            enum_name_str,
                            name
                        )));
                    }
                }

                HirType::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let struct_name_str = str_id_to_string(*struct_name);
                    let Some(def) = self.context.get_struct(&struct_name_str) else {
                        return;
                    };

                    let mut seen: std::collections::HashSet<StrId> =
                        std::collections::HashSet::new();
                    for (field_name, sub_pattern) in fields.iter() {
                        if !seen.insert(*field_name) {
                            self.record(TypeErrorKind::Generic(format!(
                                "field `{}` matched more than once in this pattern",
                                str_id_to_string(*field_name)
                            )));
                            continue;
                        }
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            self.record(TypeErrorKind::FieldNotFound {
                                struct_name: struct_name_str.clone(),
                                field: str_id_to_string(*field_name),
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
                        expected: type_to_string(scrutinee_ty),
                        found: format!("named-field pattern `{}`", name),
                    });
                }
            },

            HirPattern::Or(patterns) => {
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
                    bindings.sort_by(|(na, _), (nb, _)| {
                        str_id_to_string(*na).cmp(&str_id_to_string(*nb))
                    });

                    match &first_bindings {
                        None => first_bindings = Some(bindings),
                        Some(expected) => {
                            if !self.bindings_match(expected, &bindings) {
                                let names = |v: &[(StrId, HirType<'a, 'bump>)]| {
                                    v.iter()
                                        .map(|(n, _)| str_id_to_string(*n))
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
                    .unwrap_or(HirType::Void)
            }
            HirStmt::Block { body, span } => {
                self.set_span(*span);
                self.check_block_body(body, expected)
                    .unwrap_or(HirType::Void)
            }
            other => self.check_stmt(other).unwrap_or(HirType::Void),
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
            return HirType::Void;
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
        if matches!(scrutinee_ty, HirType::Unknown) {
            return;
        }

        let has_catch_all = arms.iter().any(|arm| {
            arm.guard.is_none()
                && matches!(arm.pattern, HirPattern::Wildcard | HirPattern::Ident(_))
        });
        if has_catch_all {
            return;
        }

        match scrutinee_ty {
            HirType::Nullable(_) => {
                let has_null = arms
                    .iter()
                    .any(|arm| arm.guard.is_none() && matches!(arm.pattern, HirPattern::Null));
                if !has_null {
                    self.record(TypeErrorKind::Generic(format!(
                        "non-exhaustive match on `{}`: missing a `null` arm",
                        type_to_string(scrutinee_ty)
                    )));
                }
            }
            HirType::Enum {
                name: enum_name, ..
            } => {
                let enum_name_str = str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let covered: std::collections::HashSet<StrId, FxHashBuilder> = arms
                    .iter()
                    .filter(|arm| arm.guard.is_none())
                    .filter_map(|arm| match &arm.pattern {
                        HirPattern::EnumVariant { variant, .. } => Some(*variant),
                        HirPattern::Struct { name, .. } => Some(*name),
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
                        enum_name_str,
                        missing.join(", ")
                    )));
                }
            }

            HirType::Boolean => {
                let mut has_true = false;
                let mut has_false = false;
                for arm in arms.iter().filter(|a| a.guard.is_none()) {
                    match &arm.pattern {
                        HirPattern::Boolean(true) => has_true = true,
                        HirPattern::Boolean(false) => has_false = true,
                        _ => {}
                    }
                }
                if !(has_true && has_false) {
                    self.record(TypeErrorKind::Generic(
                        "non-exhaustive match on `bool`: requires a wildcard (`_`) arm or both `true` and `false` arms".to_string()
                    ));
                }
            }

            HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::Usize
            | HirType::Isize
            | HirType::String
            | HirType::Char => {
                self.record(TypeErrorKind::Generic(format!(
                    "non-exhaustive match on `{}`: requires a wildcard (`_`) or binding (catch-all) arm",
                    type_to_string(scrutinee_ty)
                )));
            }

            _ => {} // structs/tuples/etc: not enforced yet
        }
    }

    pub fn join_value_types(&mut self, branches: &[HirType<'a, 'bump>]) -> HirType<'a, 'bump> {
        let mut result: Option<HirType<'a, 'bump>> = None;
        for ty in branches {
            if matches!(ty, HirType::Never) {
                continue;
            }
            match result {
                None => result = Some(*ty),
                Some(expected) => {
                    let check = self.types_compatible(&expected, ty);
                    self.recover(check, ());
                }
            }
        }
        result.unwrap_or(HirType::Never)
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
}

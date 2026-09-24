pub mod calls;
pub mod intrinsics;

use ir::{
    borrow_checker::BorrowKind,
    errors::type_error::{TypeCheckResult, TypeErrorKind},
    hir::{
        AssignmentOperator, HirExpr, HirMatchArm, HirStmt, HirType, Operator, StrId, Visibility,
    },
    ir_hasher::{FxHashMap, HashSet},
    span::SourceSpan,
};

use crate::{
    closures,
    initialization::{BareImportKind, InitNode, InitStatus, IntervalSet},
    move_state::MoveState,
    naming::{operator_symbol, str_id_to_string, type_to_string},
    type_checker::{LocalSymbolId, SymbolId},
    TypeChecker,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn check_cast_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        target_type: &HirType<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let source_type = self.check_expr(expr);
        self.check_and_record_value_use(expr, &source_type);
        let result = self.check_cast_legality(&source_type, target_type);
        self.recover(result, ());
        *target_type
    }

    pub fn check_array_literal_expr(
        &mut self,
        elements: &&[HirExpr<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        if elements.is_empty() {
            self.record(TypeErrorKind::TypeCannotBeInferred);
            return HirType::Unknown;
        }

        let first_ty = self.check_expr(&elements[0]);
        self.check_and_record_value_use(&elements[0], &first_ty);

        for elem in &elements[1..] {
            let elem_ty = self.check_expr(elem);
            self.check_and_record_value_use(elem, &elem_ty);
            let result = self.types_compatible(&first_ty, &elem_ty);
            self.recover(result, ());
        }

        HirType::Array(self.context.bump.alloc_value(first_ty), elements.len())
    }

    pub fn check_index_expr(
        &mut self,
        object: &&HirExpr<'a, 'bump>,
        index: &&HirExpr<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let object_ty = self.check_expr_suppressed(object);
        let index_ty = self.check_expr(index);

        self.recover(self.types_compatible(&HirType::I64, &index_ty), ());

        if let Some((root, path)) = self.static_field_path(object) {
            if let Some(node) = self.init_state.get(&root).cloned() {
                let target = Self::node_at_path_ref(&node, &path);
                match index {
                    HirExpr::Number(i, _) => {
                        let covered = match target {
                            InitNode::Array { ranges, .. } => ranges.contains_range(*i, *i + 1),
                            InitNode::Whole(InitStatus::Initialized) => true,
                            _ => false,
                        };
                        if !covered {
                            let root_str = str_id_to_string(root);
                            self.record(TypeErrorKind::Generic(format!(
                                "use of uninitialized value `{}[{}]`",
                                root_str, i
                            )));
                        }
                    }
                    _ => {
                        let whole_ok = match target {
                            InitNode::Whole(InitStatus::Initialized) => true,
                            InitNode::Array {
                                ranges,
                                len: Some(l),
                            } => ranges.covers_full((*l) as i64),
                            _ => false,
                        };
                        if !whole_ok {
                            let root_str = str_id_to_string(root);
                            self.record(TypeErrorKind::Generic(format!(
                            "indexing `{}` with a non-constant index requires the whole array to be initialized \
                                     (the compiler can't prove which element you're reading)",
                            root_str
                        )));
                        }
                    }
                }
            }
        }

        match object_ty {
            HirType::SafePointer { inner, .. } => {
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "indexing a raw pointer requires an unsafe block".to_string(),
                    ));
                }
                *inner
            }

            HirType::UnsafePointer { inner, .. } => {
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "indexing an unsafe pointer requires an unsafe block".to_string(),
                    ));
                }
                *inner
            }

            _ => match *Self::strip_ref(&object_ty) {
                HirType::Array(inner, _) => *inner,
                HirType::Slice(inner) => *inner,

                _ => {
                    self.record(TypeErrorKind::Generic(format!(
                        "cannot index type `{}`",
                        type_to_string(&object_ty)
                    )));
                    HirType::Unknown
                }
            },
        }
    }

    pub fn check_lambda_expr(
        &mut self,
        params: &&[ir::hir::HirLambdaParam<'a, 'bump>],
        return_type: &'a HirType<'a, 'bump>,
        body: &&HirStmt<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let mut lambda_context = self.context.create_child_scope();
        for p in *params {
            let param_name = str_id_to_string(p.name);
            let param_ty = p.param_type.unwrap_or(HirType::Unknown);
            let symbol_id = self.mint_symbol_id();
            lambda_context.add_variable(param_name, param_ty, symbol_id);
        }

        let old_context = std::mem::replace(&mut self.context, lambda_context);
        self.check_stmt(body);
        self.context = old_context;

        let param_types: Vec<HirType<'a, 'bump>> = params
            .iter()
            .map(|p| p.param_type.unwrap_or(HirType::Unknown))
            .collect();

        HirType::Lambda {
            params: self.context.bump.alloc_slice(&param_types),
            return_type: self.context.bump.alloc_value(*return_type),
        }
    }

    pub fn check_deref_expr(&mut self, expr: &&HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        let mut inner_ty = self.check_expr(expr);
        if let HirType::Nullable(inner) = inner_ty {
            if let Some((root, path)) = self.static_field_path(expr) {
                if self.is_non_null(root, &path) {
                    inner_ty = *inner;
                }
            }
        }
        if let Some(base) = self.resolve_place(expr) {
            let place = self.borrow_checker.project_deref(base);
            self.check_borrow_use(expr, place, BorrowKind::Shared);
        }
        match inner_ty {
            HirType::Ref { inner, .. } => *inner,
            HirType::SafePointer { inner, .. } => {
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "dereferencing a raw pointer requires an unsafe block".into(),
                    ));
                }

                *inner
            }

            HirType::UnsafePointer { inner, .. } => {
                if !self.in_unsafe() {
                    self.record(TypeErrorKind::Generic(
                        "dereferencing an unsafe pointer requires an unsafe block".into(),
                    ));
                }

                *inner
            }
            HirType::OwnedPointer { inner, .. } => *inner,
            _ => {
                self.record(TypeErrorKind::Generic(format!(
                    "cannot dereference non-pointer type `{}`",
                    type_to_string(&inner_ty)
                )));
                HirType::Unknown
            }
        }
    }

    pub fn check_comparison_expr(
        &mut self,
        left: &&HirExpr<'a, 'bump>,
        op: &Operator,
        right: &&HirExpr<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let left_type = self.check_expr(left);
        let right_type = self.check_expr(right);
        let result = self.check_binary_op(&left_type, op, &right_type);
        self.recover(result, HirType::Unknown)
    }

    pub fn check_assignment_expr(
        &mut self,
        target: &&HirExpr<'a, 'bump>,
        op: &AssignmentOperator,
        value: &&HirExpr<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        if let HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } =
            target
        {
            let field_name = str_id_to_string(*field);
            if field_name == "len" || field_name == "cap" {
                let obj_type = self.check_expr_suppressed(object);
                if let Some(is_owned) = Self::slice_field_owned(&obj_type) {
                    let value_type = self.check_expr(value);

                    if field_name == "cap" {
                        if !is_owned {
                            self.record(TypeErrorKind::Generic(
                                "`.cap` only exists on an owned slice".to_string(),
                            ));
                        } else {
                            self.record(TypeErrorKind::Generic(
                                "cannot assign to `.cap`: it is allocator-managed bookkeeping tied \
                                         to the slice's actual allocation size, and there is no way to write \
                                         it that keeps it in sync with the real allocation."
                                    .to_string(),
                            ));
                        }
                        return HirType::Unknown;
                    }

                    if !self.in_unsafe() {
                        self.record(TypeErrorKind::Generic(
                            "writing to `.len` requires an unsafe block: it directly edits a \
                                     slice's length bookkeeping and can expose uninitialized memory or \
                                     break drop tracking"
                                .to_string(),
                        ));
                    }
                    if !self.is_integer(&value_type) {
                        self.record(TypeErrorKind::Generic(format!(
                            "`.len` must be assigned an integer, found `{}`",
                            type_to_string(&value_type)
                        )));
                    }
                    if !matches!(op, AssignmentOperator::Assign) {
                        self.record(TypeErrorKind::Generic(
                            "`.len` only supports plain assignment, not compound assignment"
                                .to_string(),
                        ));
                    }
                    return HirType::Usize;
                }
            }
        }

        let target_type = self.check_expr_as_place(target);
        let is_uninit_value = matches!(value, HirExpr::Uninit { .. });
        let value_type = self.check_expr_expected(value, &target_type);

        if let Some(place) = self.resolve_place(target) {
            self.check_borrow_use(target, place, BorrowKind::Mutable);
            if let Some((root, path)) = self.mutation_path(target) {
                self.note_capture_use(root, &path, closures::UseLevel::Mut);
            }
            if is_uninit_value {
                self.borrow_checker.mark_place_uninit(place);
            } else {
                self.borrow_checker.mark_place_init(place);
            }
        }

        if let HirExpr::Ident(name, _) = target {
            let var_name = str_id_to_string(*name);
            if self.context.is_local_binding(&var_name) && !self.context.is_mutable(&var_name) {
                self.record(TypeErrorKind::Generic(format!(
                    "cannot assign to `{}`: it is not declared `mut`",
                    var_name
                )));
            }
        }

        if !matches!(op, AssignmentOperator::Assign) {
            self.check_read_for_compound_target(target, &target_type);
        }

        match target {
            HirExpr::Ident(name, _) => {
                if !matches!(value_type, HirType::Nullable(_)) {
                    self.mark_non_null(*name, &[]);
                } else {
                    self.clear_non_null(*name, &[]);
                }
                if is_uninit_value {
                    self.mark_whole_uninit(*name);
                } else {
                    self.mark_field_init(*name, &[]);
                }
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let Some((root, mut path)) = self.static_field_path(object) {
                    path.push(*field);
                    if !matches!(value_type, HirType::Nullable(_)) {
                        self.mark_non_null(root, &path);
                    } else {
                        self.clear_non_null(root, &path);
                    }
                    if is_uninit_value {
                        self.mark_field_uninit(root, &path);
                    } else {
                        self.mark_field_init(root, &path);
                    }
                }
            }
            HirExpr::Index { object, index, .. } => {
                if let Some((root, path)) = self.static_field_path(object) {
                    let len = match self.peek_type(object) {
                        HirType::Array(_, l) => Some(l),
                        _ => None,
                    };
                    match index {
                        HirExpr::Number(i, _) if !is_uninit_value => {
                            self.mark_array_range(root, &path, *i, *i + 1, len);
                        }
                        HirExpr::Number(i, _) => {
                            let root_node = self
                                .init_state
                                .entry(root)
                                .or_insert(InitNode::Whole(InitStatus::Uninitialized));
                            if let InitNode::Array { ranges, .. } =
                                Self::node_at_path_mut(root_node, &path)
                            {
                                let mut kept = IntervalSet::default();
                                for &(s, e) in &ranges.ranges {
                                    if e <= *i || s >= *i + 1 {
                                        kept.insert(s, e);
                                    } else {
                                        if s < *i {
                                            kept.insert(s, *i);
                                        }
                                        if *i + 1 < e {
                                            kept.insert(*i + 1, e);
                                        }
                                    }
                                }
                                *ranges = kept;
                            }
                        }
                        _ => {
                            let root_node = self
                                .init_state
                                .entry(root)
                                .or_insert(InitNode::Whole(InitStatus::Uninitialized));
                            let target_node = Self::node_at_path_mut(root_node, &path);
                            if !matches!(target_node, InitNode::Whole(InitStatus::Initialized)) {
                                *target_node = InitNode::Whole(InitStatus::Maybe);
                            }
                        }
                    }
                }
            }
            _ => {}
        }

        use ir::hir::AssignmentOperator::*;
        let bin_result = match op {
            Assign => self.types_compatible(&target_type, &value_type),
            AddAssign => self
                .check_binary_op(&target_type, &Operator::Add, &value_type)
                .map(|_| ()),
            SubtractAssign => self
                .check_binary_op(&target_type, &Operator::Subtract, &value_type)
                .map(|_| ()),
            MultiplyAssign => self
                .check_binary_op(&target_type, &Operator::Multiply, &value_type)
                .map(|_| ()),
            DivideAssign => self
                .check_binary_op(&target_type, &Operator::Divide, &value_type)
                .map(|_| ()),
            ModuloAssign => self
                .check_binary_op(&target_type, &Operator::Modulo, &value_type)
                .map(|_| ()),
            BitAndAssign => self
                .check_binary_op(&target_type, &Operator::BitAnd, &value_type)
                .map(|_| ()),
            BitOrAssign => self
                .check_binary_op(&target_type, &Operator::BitOr, &value_type)
                .map(|_| ()),
            BitXorAssign => self
                .check_binary_op(&target_type, &Operator::BitXor, &value_type)
                .map(|_| ()),
            ShiftLeftAssign => self
                .check_binary_op(&target_type, &Operator::ShiftLeft, &value_type)
                .map(|_| ()),
            ShiftRightAssign => self
                .check_binary_op(&target_type, &Operator::ShiftRight, &value_type)
                .map(|_| ()),
        };
        self.recover(bin_result, ());

        target_type
    }

    pub fn check_struct_init_expr(
        &mut self,
        name: &HirExpr<'a, 'bump>,
        args: &[ir::hir::HirFieldInit<'a, 'bump>],
        type_args: &Option<&'a [HirType<'a, 'bump>]>,
    ) -> HirType<'a, 'bump> {
        let HirExpr::Ident(struct_name_id, name_span) = name else {
            return HirType::Void;
        };
        let struct_name_str = str_id_to_string(*struct_name_id);
        let Some(ty_struct) = self.context.get_struct(&struct_name_str) else {
            self.record(TypeErrorKind::UndefinedType(struct_name_str));
            return HirType::Struct {
                name: *struct_name_id,
                field_types: &[],
                type_args: type_args.unwrap_or(&[]),
            };
        };
        self.check_bare_name_import(
            self.context.struct_owner(&struct_name_str),
            *struct_name_id,
            &struct_name_str,
            BareImportKind::Struct,
        );

        let is_generic_decl = ty_struct.generics.is_some_and(|g| !g.is_empty());

        let resolved_field_types: Vec<HirType<'a, 'bump>> = match (is_generic_decl, type_args) {
            (true, Some(ta)) => match self.instantiate_struct(*struct_name_id, ta) {
                Some(fields) => fields.to_vec(),
                None => {
                    self.record(TypeErrorKind::Generic(format!(
                        "struct `{}` expects {} type argument(s), found {}",
                        struct_name_str,
                        ty_struct.generics.map(|g| g.len()).unwrap_or(0),
                        ta.len(),
                    )));
                    ty_struct.fields.iter().map(|f| f.field_type).collect()
                }
            },
            (true, None) => match self.instantiate_struct(*struct_name_id, &[]) {
                Some(fields) => fields.to_vec(),
                None => {
                    self.record(TypeErrorKind::Generic(format!(
                    "struct `{}` is generic and requires explicit type arguments, e.g. `{}<Type> {{ .. }}`",
                    struct_name_str, struct_name_str,
                )));
                    ty_struct.fields.iter().map(|f| f.field_type).collect()
                }
            },
            (false, Some(_)) => {
                self.record(TypeErrorKind::Generic(format!(
                    "struct `{}` is not generic; no type arguments expected",
                    struct_name_str,
                )));
                ty_struct.fields.iter().map(|f| f.field_type).collect()
            }
            (false, None) => ty_struct.fields.iter().map(|f| f.field_type).collect(),
        };

        let mut seen: std::collections::HashSet<StrId> = std::collections::HashSet::new();

        for field_init in args {
            let field_name_str = str_id_to_string(field_init.name);
            let field_idx = ty_struct
                .fields
                .iter()
                .position(|f| str_id_to_string(f.name) == field_name_str);

            let Some(field_idx) = field_idx else {
                self.record(TypeErrorKind::FieldNotFound {
                    struct_name: struct_name_str.clone(),
                    field: field_name_str,
                });
                self.check_expr(&field_init.value);
                continue;
            };
            let field_type = resolved_field_types[field_idx];

            if !seen.insert(field_init.name) {
                self.record(TypeErrorKind::Generic(format!(
                    "field `{}` initialized more than once",
                    field_name_str
                )));
            }

            let arg_type = self.check_expr_expected(&field_init.value, &field_type);
            self.check_and_record_value_use(&field_init.value, &arg_type);
            let result = self.types_compatible(&field_type, &arg_type);
            self.recover(result, ());

            self.occurrences.push((
                field_init.name_span,
                field_init.name,
                field_type,
                self.context.current_module_idx,
                SymbolId::Field {
                    struct_name: *struct_name_id,
                    field_name: field_init.name,
                },
                false,
            ));
        }

        let missing: Vec<&str> = ty_struct
            .fields
            .iter()
            .filter(|f| !args.iter().any(|a| a.name == f.name))
            .map(|f| f.name.as_str())
            .collect();
        if !missing.is_empty() {
            self.record(TypeErrorKind::Generic(format!(
                "missing field(s) in struct init: {}",
                missing.join(", ")
            )));
        }

        let result_ty = HirType::Struct {
            name: *struct_name_id,
            field_types: self.context.bump.alloc_slice(&resolved_field_types),
            type_args: type_args.unwrap_or(&[]),
        };
        if let Some(owner) = self.context.struct_owner(&struct_name_str) {
            self.record_item_occurrence(*name_span, *struct_name_id, result_ty, owner);
        }
        result_ty
    }

    pub fn check_slice_expr(
        &mut self,
        object: &&HirExpr<'a, 'bump>,
        start: &&HirExpr<'a, 'bump>,
        end: &&HirExpr<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let object_ty = self.check_expr_suppressed(object);
        let start_ty = self.check_expr(start);
        let end_ty = self.check_expr(end);
        if !self.is_integer(&start_ty) || !self.is_integer(&end_ty) {
            self.record(TypeErrorKind::Generic(
                "slice bounds must be integers".to_string(),
            ));
        }
        if matches!(*Self::strip_ref(&object_ty), HirType::Array(..)) {
            self.check_slice_range_init(object, start, end);
        }
        match *Self::strip_ref(&object_ty) {
            HirType::Array(inner, _) | HirType::Slice(inner) => HirType::Slice(inner),
            _ => {
                self.record(TypeErrorKind::Generic(format!(
                    "cannot slice type `{}`",
                    type_to_string(&object_ty)
                )));
                HirType::Unknown
            }
        }
    }

    pub fn check_range_expr(
        &mut self,
        start: &&HirExpr<'a, 'bump>,
        end: &&HirExpr<'a, 'bump>,
        inclusive: &bool,
    ) -> HirType<'a, 'bump> {
        let start_ty = self.check_expr(start);
        let end_ty = self.check_expr(end);
        self.check_and_record_value_use(start, &start_ty);
        self.check_and_record_value_use(end, &end_ty);
        if !self.is_integer(&start_ty) {
            self.record(TypeErrorKind::Generic(format!(
                "range bounds must be integers, found `{}`",
                type_to_string(&start_ty)
            )));
        }
        let result = self.types_compatible(&start_ty, &end_ty);
        self.recover(result, ());
        HirType::Range {
            elem: self.context.bump.alloc_value(start_ty),
            inclusive: *inclusive,
        }
    }

    pub fn check_block_expr(
        &mut self,
        body: &&[HirStmt<'a, 'bump>],
        is_unsafe: &bool,
    ) -> HirType<'a, 'bump> {
        if *is_unsafe {
            self.unsafe_depth += 1;
        }

        self.borrow_checker.begin_scope();
        let mut block_context = self.context.create_child_scope();
        let mut value = HirType::Void;
        for stmt in *body {
            let old_context = std::mem::replace(&mut self.context, block_context);
            value = self.check_stmt(stmt).unwrap_or(HirType::Void);
            block_context = self.context.clone();
            self.context = old_context;
        }
        self.borrow_checker.end_scope();

        if *is_unsafe {
            self.unsafe_depth -= 1;
        }
        value
    }

    pub fn check_match_expr(
        &mut self,
        expr: &&HirExpr<'a, 'bump>,
        arms: &&[HirMatchArm<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        let scrutinee_ty = self.check_expr(expr);

        self.check_match_exhaustiveness(&scrutinee_ty, arms);

        let move_state_before = self.move_state.clone();
        let mut arm_types = Vec::with_capacity(arms.len());
        let mut arm_move_states = Vec::with_capacity(arms.len());

        for arm in *arms {
            let mode = self.scrutinee_binding_mode(expr);
            let scrutinee_place = self.resolve_place(expr);
            let scrutinee_provenance = self.infer_provenance(expr);
            self.binding_mode_backfill
                .insert(Self::expr_key(expr), mode);

            self.move_state = move_state_before.clone();
            self.borrow_checker.begin_scope();
            let arm_context = self.context.create_child_scope();
            let old_context = std::mem::replace(&mut self.context, arm_context);

            self.check_pattern_against_type(&arm.pattern, &scrutinee_ty);
            self.register_pattern_bindings(
                &arm.pattern,
                &scrutinee_ty,
                mode,
                scrutinee_provenance,
                scrutinee_place,
            );

            if let Some(guard) = arm.guard {
                let guard_type = self.check_expr(guard);
                if guard_type != HirType::Boolean {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: "bool".to_string(),
                        found: type_to_string(&guard_type),
                    });
                }
            }

            let arm_ty = self.check_stmt(arm.body).unwrap_or(HirType::Void);
            self.context = old_context;
            self.borrow_checker.end_scope();
            let mut bound = Vec::new();
            self.collect_pattern_bindings(&arm.pattern, &scrutinee_ty, &mut bound);
            for (name, _) in bound {
                self.local_provenance_place.remove(&name);
                self.local_ref_kind.remove(&name);
            }
            arm_move_states.push(self.move_state.clone());
            arm_types.push(arm_ty);
        }

        self.move_state = arm_move_states
            .into_iter()
            .fold(move_state_before, |acc, s| MoveState::join(&acc, &s));
        self.join_value_types(&arm_types)
    }

    pub fn check_binary_expr(
        &mut self,
        left: &&HirExpr<'a, 'bump>,
        op: &Operator,
        right: &&HirExpr<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        let left_type = self.check_expr(left);
        let right_type = self.check_expr(right);
        let result = self.check_binary_op(&left_type, op, &right_type);
        self.recover(result, HirType::Unknown)
    }

    pub fn check_ident_expr(&mut self, name: &StrId, span: &SourceSpan<'a>) -> HirType<'a, 'bump> {
        let var_name = str_id_to_string(*name);
        let (symbol_id, ty) = match self.context.get_variable(&var_name) {
            Some(ty) => ty,
            None => {
                self.record(TypeErrorKind::UndefinedVariable(var_name.clone()));
                (SymbolId::Local(LocalSymbolId(u32::MAX)), HirType::Unknown)
            }
        };
        self.check_ident_init_read(*name, &var_name, &ty);
        self.point_locals_used
            .entry(self.current_point)
            .or_default()
            .insert(*name);
        self.occurrences.push((
            *span,
            *name,
            ty,
            self.context.current_module_idx,
            symbol_id,
            false,
        ));
        ty
    }

    pub fn check_zeroed_value(&mut self, ty: &HirType<'a, 'bump>) -> HirType<'a, 'bump> {
        match ty {
            HirType::Unknown => {
                self.record(TypeErrorKind::TypeCannotBeInferred);
                HirType::Unknown
            }
            other_type => {
                if !self.is_zeroable(other_type) {
                    self.record(TypeErrorKind::Generic(format!(
                    "`undefined` cannot be used for type `{}`: it cannot be safely zero-initialized",
                    type_to_string(other_type)
                )));
                    return HirType::Unknown;
                }
                *other_type
            }
        }
    }

    pub fn check_uninit_value(&mut self, ty: &HirType<'a, 'bump>) -> HirType<'a, 'bump> {
        match ty {
            HirType::Unknown => {
                self.record(TypeErrorKind::TypeCannotBeInferred);
                HirType::Unknown
            }
            other_type => *other_type,
        }
    }

    pub fn check_enum_init_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        enum_name: &StrId,
        variant: &StrId,
        args: &[HirExpr<'a, 'bump>],
        type_args: &Option<&'bump [HirType<'a, 'bump>]>,
        span: SourceSpan<'a>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        self.set_span(span);
        let enum_name_str = str_id_to_string(*enum_name);
        let Some(enum_def) = self.context.get_enum(&enum_name_str) else {
            self.record(TypeErrorKind::UndefinedType(enum_name_str));
            return HirType::Unknown;
        };
        self.check_bare_name_import(
            self.context.enum_owner(&enum_name_str),
            *enum_name,
            &enum_name_str,
            BareImportKind::Enum,
        );

        let variant_name = str_id_to_string(*variant);
        let variant_def = enum_def
            .variants
            .iter()
            .find(|v| str_id_to_string(v.name) == variant_name);
        let Some(variant_def) = variant_def else {
            self.record(TypeErrorKind::Generic(format!(
                "enum `{}` has no variant `{}`",
                enum_name_str, variant_name
            )));
            return HirType::Unknown;
        };

        let is_generic_decl = enum_def.generics.is_some_and(|g| !g.is_empty());

        let resolved_field_types: Vec<HirType<'a, 'bump>> = match (is_generic_decl, type_args) {
            (true, Some(ta)) => match self.instantiate_enum(*enum_name, ta) {
                Some(variants) => variants
                    .iter()
                    .find(|(name, _)| *name == *variant)
                    .map(|(_, fields)| fields.to_vec())
                    .unwrap_or_else(|| variant_def.fields.iter().map(|f| f.field_type).collect()),
                None => {
                    self.record(TypeErrorKind::Generic(format!(
                        "enum `{}` expects {} type argument(s), found {}",
                        enum_name_str,
                        enum_def.generics.map(|g| g.len()).unwrap_or(0),
                        ta.len(),
                    )));
                    variant_def.fields.iter().map(|f| f.field_type).collect()
                }
            },
            (true, None) => {
                self.record(TypeErrorKind::Generic(format!(
                    "enum `{}` is generic and requires explicit type arguments, e.g. `{}<Type>::{}(..)`",
                    enum_name_str, enum_name_str, variant_name,
                )));
                variant_def.fields.iter().map(|f| f.field_type).collect()
            }
            (false, Some(_)) => {
                self.record(TypeErrorKind::Generic(format!(
                    "enum `{}` is not generic; no type arguments expected",
                    enum_name_str,
                )));
                variant_def.fields.iter().map(|f| f.field_type).collect()
            }
            (false, None) => variant_def.fields.iter().map(|f| f.field_type).collect(),
        };

        if args.len() != resolved_field_types.len() {
            self.record(TypeErrorKind::InvalidFunctionCall {
                expected_args: resolved_field_types.len(),
                found_args: args.len(),
            });
        }

        let mut arg_types: Vec<HirType<'a, 'bump>> = Vec::with_capacity(args.len());
        for (arg, field_type) in args.iter().zip(resolved_field_types.iter()) {
            let arg_type = self.check_expr_expected(arg, field_type);
            self.check_and_record_value_use(arg, &arg_type);
            self.recover(self.types_compatible(field_type, &arg_type), ());
            arg_types.push(arg_type);
        }

        if let Some(ta) = type_args {
            self.record_instance_args(expr, ta);
        }

        let final_type_args: &'bump [HirType<'a, 'bump>] = if let Some(ta) = type_args {
            ta
        } else if is_generic_decl {
            let generics = enum_def.generics.unwrap_or(&[]);
            let mut subs: FxHashMap<StrId, HirType<'a, 'bump>> = FxHashMap::default();
            for (declared_field, actual_ty) in variant_def.fields.iter().zip(arg_types.iter()) {
                self.unify_generic(&declared_field.field_type, actual_ty, &mut subs);
            }
            if let Some(HirType::Enum {
                name: exp_name,
                type_args: exp_targs,
                ..
            }) = expected
            {
                if exp_name == enum_name {
                    for (g, exp_ty) in generics.iter().zip(exp_targs.iter()) {
                        subs.entry(g.name).or_insert(*exp_ty);
                    }
                }
            }
            let inferred: Vec<HirType<'a, 'bump>> = generics
                .iter()
                .map(|g| subs.get(&g.name).copied().unwrap_or(HirType::Unknown))
                .collect();
            self.context.bump.alloc_slice_copy(&inferred)
        } else {
            &[]
        };

        debug_assert!(
            !final_type_args
                .iter()
                .any(|t| matches!(t, HirType::Unknown)),
            "enum `{}::{}` resolved with an Unknown type argument ({:?}); a variant whose fields \
             don't mention every generic parameter needs the constructor's expected type threaded \
             in via check_expr_expected/check_enum_init's `expected` param.",
            enum_name_str,
            variant_name,
            final_type_args
        );

        HirType::Enum {
            name: *enum_name,
            type_args: final_type_args,
            variants: enum_def.variants,
        }
    }

    pub fn check_field_access_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        field: StrId,
    ) -> HirType<'a, 'bump> {
        let obj_type = self.check_expr_suppressed(object);
        let mut stripped = *Self::strip_ref(&obj_type);

        if let HirType::Nullable(inner) = stripped {
            if let Some((root, path)) = self.static_field_path(object) {
                if self.is_non_null(root, &path) {
                    stripped = *inner;
                }
            }
        }

        if let HirType::Slice(_) | HirType::Array(_, _) = stripped {
            if str_id_to_string(field) == "len" {
                return HirType::Usize;
            }
        }

        let HirType::Struct {
            name: struct_name,
            type_args,
            ..
        } = stripped
        else {
            self.record(TypeErrorKind::Generic(format!(
                "Cannot access field on non-struct type: {}",
                type_to_string(&obj_type)
            )));
            return HirType::Unknown;
        };

        let struct_name_str = str_id_to_string(struct_name);
        let Some(struct_def) = self.context.get_struct(&struct_name_str) else {
            self.record(TypeErrorKind::UndefinedType(struct_name_str));
            return HirType::Unknown;
        };

        self.check_bare_name_import(
            self.context.struct_owner(&struct_name_str),
            struct_name,
            &struct_name_str,
            BareImportKind::Struct,
        );

        let field_name = str_id_to_string(field);
        let field_idx = struct_def
            .fields
            .iter()
            .position(|f| str_id_to_string(f.name) == field_name);

        let Some(field_idx) = field_idx else {
            self.record(TypeErrorKind::FieldNotFound {
                struct_name: struct_name_str,
                field: field_name,
            });
            return HirType::Unknown;
        };

        let ty = if type_args.is_empty() {
            struct_def.fields[field_idx].field_type
        } else {
            match self.instantiate_struct(struct_name, type_args) {
                Some(fields) => fields[field_idx],
                None => struct_def.fields[field_idx].field_type,
            }
        };

        self.occurrences.push((
            self.current_span,
            field,
            ty,
            self.context.current_module_idx,
            SymbolId::Field {
                struct_name,
                field_name: field,
            },
            false,
        ));

        if let Some((root, mut path)) = self.static_field_path(object) {
            path.push(field);
            let root_str = str_id_to_string(root);
            self.check_init_read_path(root, &path, &root_str);
        }

        ty
    }

    pub fn check_binary_op(
        &self,
        left: &HirType<'a, 'bump>,
        op: &Operator,
        right: &HirType<'a, 'bump>,
    ) -> TypeCheckResult<'a, HirType<'a, 'bump>> {
        use Operator::*;

        match op {
            Add | Subtract | Multiply | Divide | Modulo => {
                if self.is_numeric(left) && self.is_numeric(right) {
                    Ok(*left)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: operator_symbol(op),
                        left: type_to_string(left),
                        right: type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            Equals | NotEquals => {
                if self.is_comparable(left) && self.is_comparable(right) {
                    Ok(HirType::Boolean)
                } else if self.is_reference_like(left)
                    && self.is_reference_like(right)
                    && self.types_structurally_equal(left, right)
                {
                    // Raw pointer address comparison
                    Ok(HirType::Boolean)
                } else if self.nullable_equality_compatible(left, right) {
                    Ok(HirType::Boolean)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: operator_symbol(op),
                        left: type_to_string(left),
                        right: type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            LessThan | LessThanOrEqual | GreaterThan | GreaterThanOrEqual => {
                if self.is_comparable(left) && self.is_comparable(right) {
                    Ok(HirType::Boolean)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: operator_symbol(op),
                        left: type_to_string(left),
                        right: type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            LogicalAnd | LogicalOr => {
                if *left == HirType::Boolean && *right == HirType::Boolean {
                    Ok(HirType::Boolean)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: operator_symbol(op),
                        left: type_to_string(left),
                        right: type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight => {
                if self.is_integer(left) && self.is_integer(right) {
                    Ok(*left)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: operator_symbol(op),
                        left: type_to_string(left),
                        right: type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            _ => Err(TypeErrorKind::Generic(format!(
                "operator `{}` cannot appear in this position",
                operator_symbol(op)
            ))
            .at(self.current_span)),
        }
    }

    pub fn check_bare_name_import(
        &mut self,
        declaring_module: Option<usize>,
        name: StrId,
        name_str: &str,
        kind: BareImportKind,
    ) {
        let Some(declaring_module) = declaring_module else {
            return;
        };
        let current = self.context.current_module_idx;
        if declaring_module == current {
            return;
        }

        let explicitly_imported = self.imports_by_module.get(&current).is_some_and(|imp| {
            imp.modules.contains(&declaring_module)
                || imp.named.values().any(|&m| m == declaring_module)
        });
        if explicitly_imported {
            return;
        }

        let wildcard_modules: Vec<usize> = self
            .imports_by_module
            .get(&current)
            .map(|imp| imp.wildcard.clone())
            .unwrap_or_default();

        if !wildcard_modules.contains(&declaring_module) {
            self.record(TypeErrorKind::Generic(format!(
                "{} `{}` is declared in another module and has not been imported",
                kind.as_str(),
                name_str,
            )));
            return;
        }

        let by_module: &FxHashMap<usize, HashSet<StrId>> = match kind {
            BareImportKind::Struct => &self.structs_by_module,
            BareImportKind::Enum => &self.enums_by_module,
        };
        let candidates: Vec<usize> = wildcard_modules
            .iter()
            .copied()
            .filter(|m| by_module.get(m).is_some_and(|set| set.contains(&name)))
            .collect();

        if candidates.len() > 1 {
            let candidate_pkgs: Vec<String> = candidates
                .iter()
                .filter_map(|&m| self.context.dep_graph.borrow().get_module_package(m))
                .map(|p| p.to_string())
                .collect();
            self.record(TypeErrorKind::Generic(format!(
                "`{}` is ambiguous: it is auto-imported from multiple packages ({}); \
                 add an explicit `import` to disambiguate",
                name_str,
                candidate_pkgs.join(", "),
            )));
        }
    }

    ///   Public:   visible everywhere.
    ///   Module:   visible only within the declaring module (DOESN'T WORK NOW)
    ///   Private:  visibile only within the same file
    ///   Internal: visible anywhere in the same package, not outside it.
    pub fn check_visibility(
        &self,
        visibility: Visibility,
        declaring_module_idx: usize,
        item_kind: &str,
        item_name: &str,
    ) -> TypeCheckResult<'a, ()> {
        let visible = match visibility {
            Visibility::Public => true,
            Visibility::Private => self.context.current_module_idx == declaring_module_idx,
            Visibility::Module => {
                todo!("Implement visibility check for the module itself, similar to how Rust crates work")
            }
            Visibility::Internal => {
                let dep_graph = self.context.dep_graph.borrow();
                dep_graph.get_module_package(self.context.current_module_idx)
                    == dep_graph.get_module_package(declaring_module_idx)
            }
        };

        if visible {
            Ok(())
        } else {
            Err(TypeErrorKind::Generic(format!(
                "{} `{}` is not visible from this module",
                item_kind, item_name,
            ))
            .at(self.current_span))
        }
    }

    pub fn check_field_access_no_init_check(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        field: StrId,
    ) -> HirType<'a, 'bump> {
        let obj_type = self.check_expr_suppressed(object);
        let mut stripped = *Self::strip_ref(&obj_type);

        if let HirType::Nullable(inner) = stripped {
            if let Some((root, path)) = self.static_field_path(object) {
                if self.is_non_null(root, &path) {
                    stripped = *inner;
                }
            }
        }

        if let HirType::Slice(_) | HirType::Array(_, _) = stripped {
            if str_id_to_string(field) == "len" {
                return HirType::Usize;
            }
        }

        let HirType::Struct {
            name: struct_name,
            type_args,
            ..
        } = stripped
        else {
            self.record(TypeErrorKind::Generic(format!(
                "Cannot access field on non-struct type: {}",
                type_to_string(&obj_type)
            )));
            return HirType::Unknown;
        };

        let struct_name_str = str_id_to_string(struct_name);
        let Some(struct_def) = self.context.get_struct(&struct_name_str) else {
            self.record(TypeErrorKind::UndefinedType(struct_name_str));
            return HirType::Unknown;
        };
        self.check_bare_name_import(
            self.context.struct_owner(&struct_name_str),
            struct_name,
            &struct_name_str,
            BareImportKind::Struct,
        );

        let field_name = str_id_to_string(field);
        let Some(field_idx) = struct_def
            .fields
            .iter()
            .position(|f| str_id_to_string(f.name) == field_name)
        else {
            self.record(TypeErrorKind::FieldNotFound {
                struct_name: struct_name_str,
                field: field_name,
            });
            return HirType::Unknown;
        };

        let ty = if type_args.is_empty() {
            struct_def.fields[field_idx].field_type
        } else {
            self.instantiate_struct(struct_name, type_args)
                .map(|fields| fields[field_idx])
                .unwrap_or(struct_def.fields[field_idx].field_type)
        };

        self.occurrences.push((
            self.current_span,
            field,
            ty,
            self.context.current_module_idx,
            SymbolId::Field {
                struct_name,
                field_name: field,
            },
            false,
        ));
        ty
    }

    pub fn check_cast_legality(
        &self,
        source: &HirType<'a, 'bump>,
        target: &HirType<'a, 'bump>,
    ) -> TypeCheckResult<'a, ()> {
        if self.types_structurally_equal(source, target) {
            return Ok(());
        }

        let is_ptr = |t: &HirType<'a, 'bump>| {
            matches!(
                t,
                HirType::SafePointer { .. }
                    | HirType::UnsafePointer { .. }
                    | HirType::OwnedPointer { .. }
            )
        };

        // `void` as a pointee acts as a wildcard, like C's `void*`: any pointee
        // type may be cast to/from a pointer-to-void.
        let is_void = |t: &HirType<'a, 'bump>| matches!(t, HirType::Void);
        let pointee_compatible = |a: &HirType<'a, 'bump>, b: &HirType<'a, 'bump>| {
            is_void(a) || is_void(b) || self.types_structurally_equal(a, b)
        };

        let ok = match (source, target) {
            (s, t) if self.is_numeric(s) && self.is_numeric(t) => true,

            (HirType::Boolean, t) if self.is_numeric(t) => true,

            (
                HirType::SafePointer { inner: src, .. },
                HirType::UnsafePointer { inner: dst, .. },
            ) => pointee_compatible(src, dst),

            (
                HirType::UnsafePointer { inner: src, .. },
                HirType::SafePointer { inner: dst, .. },
            ) => pointee_compatible(src, dst),

            (HirType::SafePointer { inner: src, .. }, HirType::SafePointer { inner: dst, .. })
            | (
                HirType::UnsafePointer { inner: src, .. },
                HirType::UnsafePointer { inner: dst, .. },
            ) => pointee_compatible(src, dst),

            (HirType::Slice(src), HirType::SafePointer { inner: dst, .. }) => {
                pointee_compatible(src, dst)
            }

            (HirType::Slice(src), HirType::UnsafePointer { inner: dst, .. }) => {
                pointee_compatible(src, dst)
            }

            (HirType::Array(src, _), HirType::SafePointer { inner: dst, .. }) => {
                pointee_compatible(src, dst)
            }

            (HirType::Array(src, _), HirType::UnsafePointer { inner: dst, .. }) => {
                pointee_compatible(src, dst)
            }

            (
                HirType::OwnedPointer { inner: owned, .. },
                HirType::SafePointer { inner: dst, .. },
            ) => match owned {
                HirType::Slice(src) => pointee_compatible(src, dst),
                _ => is_void(dst),
            },

            (
                HirType::OwnedPointer { inner: owned, .. },
                HirType::UnsafePointer { inner: dst, .. },
            ) => match owned {
                HirType::Slice(src) => pointee_compatible(src, dst),
                _ => is_void(dst),
            },

            (s, t) if is_ptr(s) && self.is_integer(t) => true,
            (s, t) if self.is_integer(s) && is_ptr(t) => true,
            _ => false,
        };

        if ok {
            Ok(())
        } else {
            Err(TypeErrorKind::Generic(format!(
                "cannot cast `{}` as `{}`: no defined conversion between these types",
                type_to_string(source),
                type_to_string(target),
            ))
            .at(self.current_span))
        }
    }
}

use ir::{
    ast::MutabilityState,
    borrow_checker::{
        BorrowError, BorrowKind, Bound, IndexContainer, IndexTemplate, Interval, LoanId,
        MemoryRelation, PlaceId, ProvenanceId, ReadTemplate, RefTemplate, TemplateBase,
        TemplateProjection,
    },
    errors::type_error::TypeErrorKind,
    hir::{
        AssignmentOperator, HirEffectAccess, HirEffectSegment, HirErrorHandlerPattern, HirExpr,
        HirFunc, HirParam, HirStmt, HirType, InterpolationPart, Operator, ProvenanceAnnotation,
        ProvenancePathSegment, ProvenanceRoot, RefKind, StrId, ThisPassingKind,
    },
    ir_hasher::{FxHashMap, HashSet},
    nll_cfg::PointId,
};

use crate::{
    TypeChecker, closures,
    naming::{provenance_to_string, type_to_string},
    str_id_to_string,
    type_checker::InvalidationCause,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
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

    pub fn snapshot_call_loan_keys(&self) -> HashSet<usize> {
        self.call_loans.keys().copied().collect()
    }

    pub fn end_temp_call_loans(&mut self, before: &HashSet<usize>) {
        let new_keys: Vec<usize> = self
            .call_loans
            .keys()
            .copied()
            .filter(|k| !before.contains(k))
            .collect();
        for k in new_keys {
            if let Some(loan_id) = self.call_loans.remove(&k) {
                self.borrow_checker.end_loan_now(loan_id);
            }
        }
    }

    pub fn snapshot_closure_loan_keys(&self) -> Vec<usize> {
        self.closure_loans.keys().copied().collect()
    }

    pub fn end_temp_closure_loans(&mut self, snap: &HashSet<usize>) {
        let leaked: Vec<usize> = self
            .closure_loans
            .keys()
            .filter(|k| !snap.contains(k))
            .copied()
            .collect();
        for k in leaked {
            if let Some(ids) = self.closure_loans.remove(&k) {
                for id in ids {
                    self.borrow_checker.end_loan_now(id);
                }
            }
        }
    }

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
                let base = match self.peek_type(object) {
                    HirType::Ref { .. }
                    | HirType::SafePointer { .. }
                    | HirType::UnsafePointer { .. }
                    | HirType::OwnedPointer { .. } => self.borrow_checker.project_deref(base),
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
            } => match self.peek_type(object) {
                HirType::Array(_, _) | HirType::Slice(_) => {
                    let base = self.resolve_place(object)?;
                    let interval = self.slice_interval(start, end, *inclusive);
                    Some(self.borrow_checker.project_index(
                        base,
                        interval,
                        IndexContainer::Primitive,
                    ))
                }
                HirType::OwnedPointer { inner, .. }
                    if matches!(*inner, HirType::Slice(_) | HirType::Array(_, _)) =>
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

            HirExpr::Index { object, index, .. } => match self.peek_type(object) {
                HirType::Array(_, _) | HirType::Slice(_) => {
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
                HirType::OwnedPointer { inner, .. }
                    if matches!(*inner, HirType::Slice(_) | HirType::Array(_, _)) =>
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
                HirType::SafePointer { .. } | HirType::UnsafePointer { .. } => {
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
                    self.peek_type(left),
                    HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
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

    pub fn local_used_after(&self, point: PointId, local: StrId) -> bool {
        let mut stack: Vec<PointId> = vec![point];
        let mut visited: HashSet<PointId> = HashSet::default();

        while let Some(p) = stack.pop() {
            if !visited.insert(p) {
                continue;
            }
            if self
                .point_locals_used
                .get(&p)
                .is_some_and(|set| set.contains(&local))
            {
                return true;
            }
            if let Some(succs) = self.cfg.successors.get(&p) {
                stack.extend(succs.iter().copied());
            }
        }
        false
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
            Some(p) => format!("{} (via {})", base, provenance_to_string(p)),
            None => base.to_string(),
        }
    }

    pub fn fresh_opaque(&mut self) -> Bound {
        self.next_opaque_id += 1;
        Bound::Opaque(self.next_opaque_id)
    }

    pub fn const_eval(&self, expr: &HirExpr<'a, 'bump>) -> Option<i64> {
        match expr {
            HirExpr::Number(v, _) => Some(*v),
            HirExpr::Ident(name, _) => {
                let (sym, _) = self.context.get_variable(&str_id_to_string(*name))?;
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

    pub fn record_move(
        &mut self,
        root: StrId,
        field: Option<StrId>,
        field_ty: &HirType<'a, 'bump>,
        container_ty: Option<&HirType<'a, 'bump>>,
    ) {
        if self.copy_analysis.borrow().type_is_copy(field_ty) {
            return;
        }
        let path: Vec<ir::hir::HirEffectSegment<'bump>> = field
            .map(|f| vec![ir::hir::HirEffectSegment::Field(f)])
            .unwrap_or_default();
        self.note_capture_use(root, &path, closures::UseLevel::Move);

        if let Some(&base_place) = self.borrow_checker.local_place(root) {
            let moved_place = match field {
                None => base_place,
                Some(f) => self.borrow_checker.project_field(base_place, f),
            };
            if let Err(e) = self.borrow_checker.check_move(moved_place) {
                let path = match field {
                    None => &[][..],
                    Some(f) => &[ProvenancePathSegment::Field(f)][..],
                };
                let provenance = ProvenanceAnnotation {
                    root: ProvenanceRoot::Var(root),
                    path: self.context.bump.alloc_slice(path),
                };
                let msg = self.describe_borrow_error(&e, Some(&provenance));
                self.record(TypeErrorKind::Generic(msg));
            }
        }

        match field {
            None => self.move_state.mark_whole_moved(root),
            Some(f) => {
                let blocks_partial_move = container_ty.is_some_and(|cty| match cty {
                    HirType::Struct { name, .. } | HirType::Enum { name, .. } => {
                        self.copy_analysis.borrow().implements_drop(*name)
                    }
                    _ => false,
                });

                if blocks_partial_move {
                    self.record(TypeErrorKind::Generic(format!(
                        "cannot partially move out of `{}`, which implements `Drop`",
                        container_ty.map(|t| type_to_string(t)).unwrap_or_default()
                    )));
                    return;
                }

                self.move_state.mark_field_moved(root, f);
            }
        }
    }

    pub fn check_use(&mut self, root: StrId, field: Option<StrId>, root_ty: &HirType<'a, 'bump>) {
        if self.copy_analysis.borrow().type_is_copy(root_ty) {
            return;
        }
        let name = str_id_to_string(root);
        match field {
            None => {
                if self.move_state.blocks_whole_use(root) {
                    self.record(TypeErrorKind::Generic(format!(
                        "use of moved value: `{}`",
                        name
                    )));
                }
            }
            Some(f) => {
                if self.move_state.is_field_moved(root, f) {
                    self.record(TypeErrorKind::Generic(format!(
                        "use of moved value: `{}.{}`",
                        name,
                        str_id_to_string(f)
                    )));
                }
            }
        }
    }

    pub fn check_and_record_value_use(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ty: &HirType<'a, 'bump>,
    ) {
        if self.is_static_place(expr) {
            return;
        }

        if let Some((root, path)) = self.effect_path_of(expr) {
            self.note_capture_use(root, &path, closures::UseLevel::Read);
        }
        if let Some(place) = self.resolve_place(expr) {
            self.check_borrow_use(expr, place, BorrowKind::Shared);
        }
        self.check_no_move_out_of_borrow(expr, ty);
        match expr {
            HirExpr::Ident(name, _) => {
                self.check_use(*name, None, ty);
                self.record_move(*name, None, ty, None);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let HirExpr::Ident(root, _) = &**object {
                    let root_name = str_id_to_string(*root);
                    let container_ty = self.context.get_variable(&root_name);
                    self.check_use(*root, Some(*field), ty);
                    self.record_move(*root, Some(*field), ty, container_ty.map(|f| f.1).as_ref());
                }
            }
            _ => {}
        }
    }

    /// A by-value read of a non-Copy value out of a place we don't own leaves two
    /// owners (the old place and the new binding). In safe code that is an error;
    /// unsafe code (stdlib internals) takes explicit responsibility.
    fn check_no_move_out_of_borrow(&mut self, expr: &HirExpr<'a, 'bump>, ty: &HirType<'a, 'bump>) {
        if self.in_unsafe() {
            return;
        }
        if !matches!(
            expr,
            HirExpr::Deref { .. }
                | HirExpr::Index { .. }
                | HirExpr::FieldAccess { .. }
                | HirExpr::Get { .. }
        ) {
            return;
        }
        if self.copy_analysis.borrow().type_is_copy(ty) {
            return;
        }
        if self.place_is_behind_borrow(expr) {
            self.record(TypeErrorKind::Generic(format!(
                "cannot move a value of type `{}` out of a reference, pointer or slice; \
                 it is not `Copy`. Use `$replace` to take it explicitly",
                type_to_string(ty)
            )));
        }
    }

    pub fn check_potential_this_param_for_move(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
        func: HirFunc<'a, 'bump>,
        params: &[HirParam<'a, 'bump>],
        ret_ty: HirType<'a, 'bump>,
    ) -> Option<HirType<'a, 'bump>> {
        if let Some(this_param) = params.first() {
            if matches!(this_param, HirParam::This { .. }) {
                self.record(TypeErrorKind::IllegalThisParam {
                    func_name: func.unmangled_name.to_string(),
                });
                return Some(ret_ty);
            }

            let template = if self.return_type_may_alias(&ret_ty) {
                Some(self.analyze_ref_template(&func))
            } else {
                None
            };

            let templated_base_param = match &template {
                Some(RefTemplate::Path {
                    base: TemplateBase::Param(i),
                    ..
                }) => Some(*i),
                _ => None,
            };

            let read_templates = self.analyze_read_templates(&func);
            for (arg_idx, arg) in args.iter().enumerate() {
                if let HirExpr::Ref {
                    expr: inner,
                    ref_kind: RefKind::Shared,
                    ..
                } = arg
                {
                    if let Some(template) = read_templates.get(arg_idx) {
                        self.check_call_arg_read_effects(inner, template, args);
                    }
                }
            }

            let arg_loans =
                self.check_all_func_args(args, params, templated_base_param, Some(func));
            self.apply_callee_concurrency(&func, args);

            if let Some(loan_id) =
                self.finalize_call_loans(None, args, arg_loans, &ret_ty, template)
            {
                self.call_loans.insert(Self::expr_key(expr), loan_id);
            }

            return Some(ret_ty);
        }
        Some(ret_ty)
    }

    pub fn finalize_call_loans(
        &mut self,
        receiver: Option<&HirExpr<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
        arg_loans: Vec<LoanId>,
        ret_ty: &HirType<'a, 'bump>,
        template: Option<RefTemplate>,
    ) -> Option<LoanId> {
        let Some(template) = template else {
            for loan in arg_loans {
                self.borrow_checker.end_loan_now(loan);
            }
            return None;
        };

        if matches!(template, RefTemplate::Path { .. }) {
            for &loan in &arg_loans {
                self.borrow_checker.end_loan_now(loan);
            }
        }

        let place = self.resolve_template_place(&template, receiver, args)?;

        let result = if matches!(
            ret_ty,
            HirType::Ref {
                ref_kind: RefKind::Unique,
                ..
            }
        ) {
            self.borrow_checker.borrow_mut(place)
        } else if matches!(
            ret_ty,
            HirType::Ref {
                ref_kind: RefKind::Alias,
                ..
            }
        ) {
            self.borrow_checker.borrow_alias(place)
        } else {
            self.borrow_checker.borrow_shared(place)
        };

        match result {
            Ok(loan_id) => Some(loan_id),
            Err(e) => {
                let provenance = self.provenance_from_template(&template, receiver, args);
                let msg = self.describe_borrow_error(&e, provenance.as_ref());
                self.record(TypeErrorKind::Generic(msg));
                None
            }
        }
    }

    pub fn analyze_read_templates(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<ReadTemplate> {
        if let Some(t) = self.read_templates.get(&func.name) {
            return t.clone();
        }
        self.read_templates.insert(func.name, Vec::new());
        let templates = self.build_read_templates(func);
        self.read_templates.insert(func.name, templates.clone());
        templates
    }

    pub fn build_read_templates(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<ReadTemplate> {
        let Some(params) = func.params else {
            return Vec::new();
        };

        let mut param_index: FxHashMap<StrId, usize> = FxHashMap::default();
        let mut has_this = false;
        let mut normal_idx = 0usize;
        for p in params.iter() {
            match p {
                HirParam::Normal { name, .. } => {
                    param_index.insert(*name, normal_idx);
                    normal_idx += 1;
                }
                HirParam::This { .. } => has_this = true,
            }
        }

        let mut templates = vec![ReadTemplate::Paths(Vec::new()); normal_idx];
        if let Some(body) = func.body {
            let prev_module = std::mem::replace(
                &mut self.context.current_module_idx,
                func.declaring_module_idx,
            );
            self.collect_param_reads_stmt(&body, &param_index, has_this, &mut templates);
            self.context.current_module_idx = prev_module;
        }
        templates
    }

    pub fn record_param_read(
        base: TemplateBase,
        projections: Vec<TemplateProjection>,
        templates: &mut [ReadTemplate],
    ) {
        let TemplateBase::Param(i) = base else {
            return;
        };
        let Some(slot) = templates.get_mut(i) else {
            return;
        };
        if projections.is_empty() {
            *slot = ReadTemplate::Opaque;
        } else if let ReadTemplate::Paths(paths) = slot {
            paths.push(projections);
        }
    }

    pub fn collect_param_reads_write_target(
        &mut self,
        target: &HirExpr<'a, 'bump>,
        param_index: &FxHashMap<StrId, usize>,
        has_this: bool,
        templates: &mut [ReadTemplate],
    ) {
        if let Some((base, mut projections)) = Self::expr_to_template(target, param_index, has_this)
        {
            projections.pop();
            if !projections.is_empty() {
                Self::record_param_read(base, projections, templates);
            }
            return;
        }
        self.collect_param_reads_expr(target, param_index, has_this, templates);
    }

    pub fn mut_raw_ptr_cast_operand<'e>(
        expr: &'e HirExpr<'a, 'bump>,
    ) -> Option<&'e HirExpr<'a, 'bump>> {
        let HirExpr::Cast {
            expr: inner,
            target_type,
            ..
        } = expr
        else {
            return None;
        };
        match target_type {
            HirType::UnsafePointer {
                mutability_state, ..
            }
            | HirType::SafePointer {
                mutability_state, ..
            } if *mutability_state == MutabilityState::Mut => Some(inner),
            _ => None,
        }
    }

    pub fn resolve_callee_for_template(
        &self,
        callee: &HirExpr<'a, 'bump>,
    ) -> Option<HirFunc<'a, 'bump>> {
        match callee {
            HirExpr::Ident(name, _) => self.context.get_function(&str_id_to_string(*name)),
            HirExpr::ModuleAccess(access) => {
                let cur = self.context.current_module_idx;
                let aliased: Option<usize> = if access.path.len() == 1 {
                    self.imports_by_module.get(&cur).and_then(|imp| {
                        imp.named
                            .get(&access.path[0])
                            .or_else(|| imp.module_aliases.get(&access.path[0]))
                            .copied()
                    })
                } else {
                    None
                };
                let midx = match aliased {
                    Some(m) => m,
                    None => self
                        .context
                        .dep_graph
                        .borrow()
                        .resolve_module_path(access.path)?,
                };
                self.context
                    .get_module_function(midx, &access.member.to_string())
            }
            // Methods, interface calls, lambdas: no receiver type here.
            _ => None,
        }
    }

    /// Whether the callee may read through its `arg_idx`th parameter's pointee.
    /// Bodiless (extern/FFI) callees are trusted; callees with a body use their own
    /// read template; anything unresolvable is assumed to read.
    pub fn callee_may_read_arg(&mut self, callee: &HirExpr<'a, 'bump>, arg_idx: usize) -> bool {
        let Some(func) = self.resolve_callee_for_template(callee) else {
            return true;
        };
        let Some(params) = func.params else {
            return true;
        };
        // Template indices count only `Normal` params, so bail out if `this` shifts them.
        if arg_idx >= params.len() || params.iter().any(|p| matches!(p, HirParam::This { .. })) {
            return true;
        }
        if func.body.is_none() {
            return false;
        }
        let templates = self.analyze_read_templates(&func);
        templates
            .get(arg_idx)
            .map_or(true, Self::read_template_touches_contents)
    }

    pub fn collect_param_reads_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        param_index: &FxHashMap<StrId, usize>,
        has_this: bool,
        templates: &mut [ReadTemplate],
    ) {
        if let Some((base, projections)) = Self::expr_to_template(expr, param_index, has_this) {
            Self::record_param_read(base, projections, templates);
            return;
        }

        match expr {
            HirExpr::Match { expr, arms, .. } => {
                self.collect_param_reads_expr(expr, param_index, has_this, templates);
                for arm in arms.iter() {
                    if let Some(guard) = arm.guard {
                        self.collect_param_reads_expr(guard, param_index, has_this, templates);
                    }
                    self.collect_param_reads_stmt(arm.body, param_index, has_this, templates);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.collect_param_reads_stmt(s, param_index, has_this, templates);
                }
            }
            HirExpr::Range { start, end, .. } => {
                self.collect_param_reads_expr(start, param_index, has_this, templates);
                self.collect_param_reads_expr(end, param_index, has_this, templates);
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.collect_param_reads_expr(object, param_index, has_this, templates);
                self.collect_param_reads_expr(start, param_index, has_this, templates);
                self.collect_param_reads_expr(end, param_index, has_this, templates);
            }
            HirExpr::Tuple(exprs, _)
            | HirExpr::ArrayLiteral {
                elements: exprs, ..
            } => {
                for e in exprs.iter() {
                    self.collect_param_reads_expr(e, param_index, has_this, templates);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.collect_param_reads_expr(left, param_index, has_this, templates);
                self.collect_param_reads_expr(right, param_index, has_this, templates);
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                self.collect_param_reads_expr(callee, param_index, has_this, templates);
                for (idx, a) in args.iter().enumerate() {
                    if let Some(inner) = Self::mut_raw_ptr_cast_operand(a) {
                        if !self.callee_may_read_arg(callee, idx) {
                            self.collect_param_reads_write_target(
                                inner,
                                param_index,
                                has_this,
                                templates,
                            );
                            continue;
                        }
                    }
                    self.collect_param_reads_expr(a, param_index, has_this, templates);
                }
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.collect_param_reads_expr(object, param_index, has_this, templates);
            }
            HirExpr::Assignment {
                target, op, value, ..
            } => {
                if matches!(op, AssignmentOperator::Assign) {
                    self.collect_param_reads_write_target(target, param_index, has_this, templates);
                } else {
                    // compound assignment reads the old value
                    self.collect_param_reads_expr(target, param_index, has_this, templates);
                }
                self.collect_param_reads_expr(value, param_index, has_this, templates);
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.collect_param_reads_expr(&f.value, param_index, has_this, templates);
                }
            }
            HirExpr::EnumInit { args, .. } => {
                for a in args.iter() {
                    self.collect_param_reads_expr(a, param_index, has_this, templates);
                }
            }
            HirExpr::ExprList { list, .. } => {
                for e in list.iter() {
                    self.collect_param_reads_expr(e, param_index, has_this, templates);
                }
            }
            HirExpr::Deref { expr, .. }
            | HirExpr::Cast { expr, .. }
            | HirExpr::Ref { expr, .. } => {
                self.collect_param_reads_expr(expr, param_index, has_this, templates);
            }
            HirExpr::Index { object, index, .. } => {
                self.collect_param_reads_expr(object, param_index, has_this, templates);
                self.collect_param_reads_expr(index, param_index, has_this, templates);
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.collect_param_reads_expr(e, param_index, has_this, templates);
                    }
                }
            }
            HirExpr::If { if_stmt, .. } => {
                self.collect_param_reads_stmt(if_stmt, param_index, has_this, templates);
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.collect_param_reads_expr(a, param_index, has_this, templates);
                }
            }
            HirExpr::OrElse {
                value, else_body, ..
            } => {
                self.collect_param_reads_expr(value, param_index, has_this, templates);
                for s in else_body.iter() {
                    self.collect_param_reads_stmt(s, param_index, has_this, templates);
                }
            }
            HirExpr::Lambda { .. } => {
                // TODO: not descending
                // into closure bodies means a captured parameter used
                // inside one is silently treated as unread rather than
                // Opaque.
            }
            // Ident/This/ModuleAccess/GenericIdent/literals/Undefined/
            // UnknownIntrinsic: either already handled by the
            // expr_to_template attempt above, or carry no sub-expressions.
            _ => {}
        }
    }

    pub fn collect_param_reads_stmt(
        &mut self,
        stmt: &HirStmt<'a, 'bump>,
        param_index: &FxHashMap<StrId, usize>,
        has_this: bool,
        templates: &mut [ReadTemplate],
    ) {
        match stmt {
            HirStmt::Let {
                value,
                catch_pattern,
                ..
            } => {
                self.collect_param_reads_expr(value, param_index, has_this, templates);
                if let Some(pattern) = catch_pattern {
                    match pattern {
                        HirErrorHandlerPattern::Single { body, .. } => {
                            for s in body.iter() {
                                self.collect_param_reads_stmt(s, param_index, has_this, templates);
                            }
                        }
                        HirErrorHandlerPattern::Multiple { branches } => {
                            for branch in branches.iter() {
                                for s in branch.body.iter() {
                                    self.collect_param_reads_stmt(
                                        s,
                                        param_index,
                                        has_this,
                                        templates,
                                    );
                                }
                            }
                        }
                    }
                }
            }
            HirStmt::Const(c) => {
                self.collect_param_reads_expr(&c.value, param_index, has_this, templates)
            }
            HirStmt::Return(Some(e), _) | HirStmt::Break(Some(e), _) => {
                self.collect_param_reads_expr(e, param_index, has_this, templates)
            }
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => {}
            HirStmt::Expr(e) => self.collect_param_reads_expr(e, param_index, has_this, templates),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.collect_param_reads_expr(cond, param_index, has_this, templates);
                for s in then_block.iter() {
                    self.collect_param_reads_stmt(s, param_index, has_this, templates);
                }
                if let Some(e) = else_block {
                    self.collect_param_reads_stmt(e, param_index, has_this, templates);
                }
            }
            HirStmt::While { cond, body } => {
                self.collect_param_reads_expr(cond, param_index, has_this, templates);
                self.collect_param_reads_stmt(body, param_index, has_this, templates);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.collect_param_reads_stmt(i, param_index, has_this, templates);
                }
                if let Some(c) = condition {
                    self.collect_param_reads_expr(c, param_index, has_this, templates);
                }
                if let Some(inc) = increment {
                    self.collect_param_reads_expr(inc, param_index, has_this, templates);
                }
                self.collect_param_reads_stmt(body, param_index, has_this, templates);
            }
            HirStmt::Block { body, span: _ } => {
                for s in body.iter() {
                    self.collect_param_reads_stmt(s, param_index, has_this, templates);
                }
            }
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.collect_param_reads_expr(expr, param_index, has_this, templates);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_param_reads_expr(g, param_index, has_this, templates);
                    }
                    self.collect_param_reads_stmt(arm.body, param_index, has_this, templates);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                self.collect_param_reads_stmt(body, param_index, has_this, templates)
            }
        }
    }

    pub fn resolve_template_place_from(
        &mut self,
        mut place: PlaceId,
        projections: &[TemplateProjection],
        call_args: &[HirExpr<'a, 'bump>],
    ) -> PlaceId {
        for proj in projections {
            place = match proj {
                TemplateProjection::Field(f) => self.borrow_checker.project_field(place, *f),
                TemplateProjection::Deref => self.borrow_checker.project_deref(place),
                TemplateProjection::Index(idx_template) => {
                    let bound = match idx_template {
                        IndexTemplate::Const(c) => Bound::Const(*c),
                        IndexTemplate::Param(j) => match call_args.get(*j) {
                            Some(a) => self.expr_to_bound(a),
                            None => return place,
                        },
                        IndexTemplate::Opaque => return place,
                    };
                    let interval = Interval {
                        lower: bound.clone(),
                        upper: bound,
                    };
                    self.borrow_checker
                        .project_index(place, interval, IndexContainer::Primitive)
                }
            };
        }
        place
    }

    pub fn read_template_touches_contents(template: &ReadTemplate) -> bool {
        match template {
            ReadTemplate::Opaque => true,
            ReadTemplate::Paths(paths) => paths.iter().any(|projections| {
                projections
                    .iter()
                    .any(|p| matches!(p, TemplateProjection::Index(_) | TemplateProjection::Deref))
            }),
        }
    }

    pub fn check_call_arg_read_effects(
        &mut self,
        inner: &HirExpr<'a, 'bump>,
        read_template: &ReadTemplate,
        call_args: &[HirExpr<'a, 'bump>],
    ) {
        let Some(base_place) = self.resolve_place(inner) else {
            return;
        };
        let Some(&root) = self.borrow_checker.place_roots.get(&base_place) else {
            return;
        };
        let Some(loan_ids) = self.borrow_checker.root_loans.get(&root).cloned() else {
            return;
        };
        if loan_ids.is_empty() {
            return;
        }

        match read_template {
            ReadTemplate::Opaque => {
                for loan_id in &loan_ids {
                    let Some(loan) = self.borrow_checker.active_loans.get(loan_id) else {
                        continue;
                    };
                    if loan.kind != BorrowKind::Mutable {
                        continue;
                    }
                    if let Ok(MemoryRelation::Overlap) =
                        self.borrow_checker.overlaps(base_place, loan.place)
                    {
                        self.record(TypeErrorKind::Generic(
                            "cannot pass this reference here: it may alias a value that's still \
                             mutably borrowed, and this call's effect on it isn't provably disjoint \
                             (the callee's parameter usage couldn't be bounded)"
                                .to_string(),
                        ));
                    }
                }
            }
            ReadTemplate::Paths(paths) => {
                for projections in paths {
                    let read_place =
                        self.resolve_template_place_from(base_place, projections, call_args);
                    for loan_id in &loan_ids {
                        let Some(loan) = self.borrow_checker.active_loans.get(loan_id) else {
                            continue;
                        };
                        if loan.kind != BorrowKind::Mutable {
                            continue;
                        }
                        if let Ok(MemoryRelation::Overlap) =
                            self.borrow_checker.overlaps(read_place, loan.place)
                        {
                            self.record(TypeErrorKind::Generic(
                                "cannot pass this reference here: the callee reads a part of it \
                                 that's still mutably borrowed"
                                    .to_string(),
                            ));
                        }
                    }
                }
            }
        }
    }

    pub fn in_unsafe(&self) -> bool {
        self.unsafe_depth != 0
    }

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

    pub fn expr_references_local(&self, expr: &HirExpr<'a, 'bump>, local: StrId) -> bool {
        match expr {
            HirExpr::Match { expr, arms, .. } => {
                self.expr_references_local(expr, local)
                    || arms.iter().any(|arm| {
                        arm.guard.is_some_and(|g| self.expr_references_local(g, local))
                            || self.stmt_references_local(arm.body, local)
                    })
            }
            HirExpr::Block { body, .. } => body.iter().any(|s| self.stmt_references_local(s, local)),
            HirExpr::Range { start, end, .. } => {
                self.expr_references_local(start, local) || self.expr_references_local(end, local)
            }
            HirExpr::Slice { object, start, end, .. } => {
                self.expr_references_local(object, local)
                    || self.expr_references_local(start, local)
                    || self.expr_references_local(end, local)
            }
            HirExpr::Ident(name, _) => *name == local,
            HirExpr::Tuple(exprs, _) | HirExpr::ArrayLiteral { elements: exprs, .. } =>
                exprs.iter().any(|e| self.expr_references_local(e, local)),
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } =>
                self.expr_references_local(left, local) || self.expr_references_local(right, local),
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } =>
                self.expr_references_local(callee, local) || args.iter().any(|a| self.expr_references_local(a, local)),
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } =>
                self.expr_references_local(object, local),
            HirExpr::Assignment { target, value, .. } =>
                self.expr_references_local(target, local) || self.expr_references_local(value, local),
            HirExpr::StructInit { args, .. } => args.iter().any(|f| self.expr_references_local(&f.value, local)),
            HirExpr::EnumInit { args, .. } => args.iter().any(|a| self.expr_references_local(a, local)),
            HirExpr::ExprList { list, .. } => list.iter().any(|e| self.expr_references_local(e, local)),
            HirExpr::Deref { expr, .. } | HirExpr::Ref { expr, .. } | HirExpr::Cast { expr, .. } =>
                self.expr_references_local(expr, local),
            HirExpr::Index { object, index, .. } =>
                self.expr_references_local(object, local) || self.expr_references_local(index, local),
            HirExpr::Lambda { body, .. } => self.stmt_references_local(body, local),
            HirExpr::InterpolatedString(parts) => parts.iter().any(|p| {
                matches!(p, ir::hir::InterpolationPart::Expr(e) if self.expr_references_local(e, local))
            }),
            HirExpr::This { .. } | HirExpr::ModuleAccess(_) | HirExpr::GenericIdent(..)
            | HirExpr::Number(..) | HirExpr::Decimal(..) | HirExpr::String(..)
            | HirExpr::Boolean(..) | HirExpr::Null(_) | HirExpr::Undefined { .. } | HirExpr::Uninit { .. } | HirExpr::Char(_, _) => false,
            HirExpr::Intrinsic { args, .. } => {
                args.iter().any(|a| self.expr_references_local(a, local))
            }
            HirExpr::UnknownIntrinsic { .. } => unimplemented!(),
            HirExpr::If { if_stmt, span: _ } => self.stmt_references_local(*if_stmt, local),
            HirExpr::OrElse { value, else_body, .. } => {
                self.expr_references_local(value, local) || else_body.iter().any(|b| self.stmt_references_local(b, local))
            },
        }
    }

    pub fn stmt_references_local(&self, stmt: &HirStmt<'a, 'bump>, local: StrId) -> bool {
        match stmt {
            HirStmt::Let { value, .. } => self.expr_references_local(value, local),
            HirStmt::Return(Some(e), _) | HirStmt::Break(Some(e), _) => {
                self.expr_references_local(e, local)
            }
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => false,
            HirStmt::Expr(e) => self.expr_references_local(e, local),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.expr_references_local(cond, local)
                    || then_block
                        .iter()
                        .any(|s| self.stmt_references_local(s, local))
                    || else_block.is_some_and(|s| self.stmt_references_local(s, local))
            }
            HirStmt::While { cond, body } => {
                self.expr_references_local(cond, local) || self.stmt_references_local(body, local)
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                init.is_some_and(|s| self.stmt_references_local(s, local))
                    || condition.is_some_and(|c| self.expr_references_local(c, local))
                    || increment.is_some_and(|c| self.expr_references_local(c, local))
                    || self.stmt_references_local(body, local)
            }
            HirStmt::Block { body, span: _ } => {
                body.iter().any(|s| self.stmt_references_local(s, local))
            }
            HirStmt::Const(c) => self.expr_references_local(&c.value, local),
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.expr_references_local(expr, local)
                    || arms.iter().any(|arm| {
                        arm.guard
                            .is_some_and(|g| self.expr_references_local(g, local))
                            || self.stmt_references_local(arm.body, local)
                    })
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                self.stmt_references_local(body, local)
            }
        }
    }

    pub fn collect_locals_used_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        if let Some(&point) = self.stmt_points.get(&Self::stmt_key(stmt)) {
            self.current_point = point;
        }
        match stmt {
            HirStmt::Let {
                value,
                catch_pattern,
                ..
            } => {
                self.collect_locals_used_expr(value);
                if let Some(pattern) = catch_pattern {
                    match pattern {
                        HirErrorHandlerPattern::Single { body, .. } => {
                            for s in body.iter() {
                                self.collect_locals_used_stmt(s);
                            }
                        }
                        HirErrorHandlerPattern::Multiple { branches } => {
                            for branch in branches.iter() {
                                for s in branch.body.iter() {
                                    self.collect_locals_used_stmt(s);
                                }
                            }
                        }
                    }
                }
            }
            HirStmt::Return(Some(e), _) | HirStmt::Break(Some(e), _) => {
                self.collect_locals_used_expr(e);
            }
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => {}
            HirStmt::Expr(e) => self.collect_locals_used_expr(e),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.collect_locals_used_expr(cond);
                for s in then_block.iter() {
                    self.collect_locals_used_stmt(s);
                }
                if let Some(e) = else_block {
                    self.collect_locals_used_stmt(e);
                }
            }
            HirStmt::While { cond, body } => {
                self.collect_locals_used_expr(cond);
                self.collect_locals_used_stmt(body);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.collect_locals_used_stmt(i);
                }
                if let Some(c) = condition {
                    self.collect_locals_used_expr(c);
                }
                if let Some(inc) = increment {
                    self.collect_locals_used_expr(inc);
                }
                self.collect_locals_used_stmt(body);
            }
            HirStmt::Block { body, span: _ } => {
                for s in body.iter() {
                    self.collect_locals_used_stmt(s);
                }
            }
            HirStmt::Const(c) => self.collect_locals_used_expr(&c.value),
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.collect_locals_used_expr(expr);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_locals_used_expr(g);
                    }
                    self.collect_locals_used_stmt(arm.body);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                self.collect_locals_used_stmt(body);
            }
        }
    }

    pub fn collect_locals_used_expr(&mut self, expr: &HirExpr<'a, 'bump>) {
        match expr {
            HirExpr::Ident(name, _) => {
                self.point_locals_used
                    .entry(self.current_point)
                    .or_default()
                    .insert(*name);
            }
            HirExpr::Match { expr, arms, .. } => {
                self.collect_locals_used_expr(expr);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_locals_used_expr(g);
                    }
                    self.collect_locals_used_stmt(arm.body);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.collect_locals_used_stmt(s);
                }
            }
            HirExpr::Range { start, end, .. } => {
                self.collect_locals_used_expr(start);
                self.collect_locals_used_expr(end);
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.collect_locals_used_expr(object);
                self.collect_locals_used_expr(start);
                self.collect_locals_used_expr(end);
            }
            HirExpr::Tuple(exprs, _)
            | HirExpr::ArrayLiteral {
                elements: exprs, ..
            } => {
                for e in exprs.iter() {
                    self.collect_locals_used_expr(e);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.collect_locals_used_expr(left);
                self.collect_locals_used_expr(right);
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                self.collect_locals_used_expr(callee);
                for a in args.iter() {
                    self.collect_locals_used_expr(a);
                }
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.collect_locals_used_expr(object);
            }
            HirExpr::Assignment { target, value, .. } => {
                self.collect_locals_used_expr(target);
                self.collect_locals_used_expr(value);
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.collect_locals_used_expr(&f.value);
                }
            }
            HirExpr::EnumInit { args, .. } => {
                for a in args.iter() {
                    self.collect_locals_used_expr(a);
                }
            }
            HirExpr::ExprList { list, .. } => {
                for e in list.iter() {
                    self.collect_locals_used_expr(e);
                }
            }
            HirExpr::Deref { expr, .. }
            | HirExpr::Cast { expr, .. }
            | HirExpr::Ref { expr, .. } => {
                self.collect_locals_used_expr(expr);
            }
            HirExpr::Index { object, index, .. } => {
                self.collect_locals_used_expr(object);
                self.collect_locals_used_expr(index);
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.collect_locals_used_expr(e);
                    }
                }
            }
            HirExpr::If { if_stmt, .. } => {
                self.collect_locals_used_stmt(if_stmt);
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.collect_locals_used_expr(a);
                }
            }
            HirExpr::Lambda { body, .. } => {
                self.collect_locals_used_stmt(body);
            }
            HirExpr::This { .. }
            | HirExpr::ModuleAccess(_)
            | HirExpr::GenericIdent(..)
            | HirExpr::Number(..)
            | HirExpr::Decimal(..)
            | HirExpr::String(..)
            | HirExpr::Boolean(..)
            | HirExpr::Null(_)
            | HirExpr::Undefined { .. }
            | HirExpr::Uninit { .. }
            | HirExpr::Char(_, _) => {}
            HirExpr::UnknownIntrinsic { .. } => {}
            HirExpr::OrElse {
                value, else_body, ..
            } => {
                self.collect_locals_used_expr(value);
                for stmt in *else_body {
                    self.collect_locals_used_stmt(stmt);
                }
            }
        }
    }

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

    pub fn place_is_within(&self, place: PlaceId, ancestor: PlaceId) -> bool {
        let mut cur = Some(place);
        while let Some(p) = cur {
            if p == ancestor {
                return true;
            }
            cur = self.borrow_checker.places.get(&p).and_then(|pl| pl.parent);
        }
        false
    }

    /// Every live loan on `place` or anything beneath it now points at freed memory.
    pub fn invalidate_loans_under(&mut self, place: PlaceId, cause: InvalidationCause) {
        let hit: Vec<ProvenanceId> = self
            .borrow_checker
            .active_loans
            .values()
            .filter(|l| self.place_is_within(l.place, place))
            .map(|l| l.provenance_id)
            .collect();
        for p in hit {
            self.invalidated_provenances.insert(p, cause);
        }
    }

    pub fn invalidating_call_for(&self, prov: ProvenanceId) -> Option<InvalidationCause> {
        let mut stack = vec![prov];
        let mut seen: HashSet<ProvenanceId> = HashSet::default();
        while let Some(p) = stack.pop() {
            if !seen.insert(p) {
                continue;
            }
            if let Some(&c) = self.invalidated_provenances.get(&p) {
                return Some(c);
            }
            if let Some(pr) = self.borrow_checker.provenances.get(&p) {
                stack.extend(pr.parents.iter().copied());
            }
        }
        None
    }

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
                let prefix = path[..path.len() - 1].to_vec();
                if matches!(
                    self.field_type_at(owner, &prefix),
                    Some(HirType::OwnedPointer { inner, .. }) if matches!(*inner, HirType::Slice(_))
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
        let Some(base) = self.resolve_place(receiver) else {
            return;
        };
        for path in paths {
            let mut place = base;
            for f in &path {
                place = self.borrow_checker.project_field(place, *f);
            }
            self.borrow_checker.reallocate_place(place);
        }
    }

    pub(crate) fn propagate_invalidations(&mut self) {
        loop {
            let mut changed = false;
            let edges: Vec<(StrId, Vec<StrId>)> = self
                .fn_this_calls
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect();
            for (caller, callees) in edges {
                for callee in callees {
                    let Some(paths) = self.fn_invalidates.get(&callee).cloned() else {
                        continue;
                    };
                    let entry = self.fn_invalidates.entry(caller).or_default();
                    for p in paths {
                        if !entry.contains(&p) {
                            entry.push(p);
                            changed = true;
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
    }

    fn is_pointer_like(ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::Ref { .. } | HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
        )
    }

    /// Does `expr` name a place that lives behind a reference or pointer, directly or
    /// through owned field/index projections?
    fn place_is_behind_borrow(&mut self, expr: &HirExpr<'a, 'bump>) -> bool {
        match expr {
            HirExpr::Ident(name, _) => self.local_ref_kind.contains_key(name),
            HirExpr::This { .. } => self.local_ref_kind.contains_key(&self.this_id),
            HirExpr::Deref { expr: inner, .. } => {
                Self::is_pointer_like(&self.peek_type(inner)) || self.place_is_behind_borrow(inner)
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                Self::is_pointer_like(&self.peek_type(object))
                    || self.place_is_behind_borrow(object)
            }
            HirExpr::Index { object, .. } => {
                let oty = self.peek_type(object);
                Self::is_pointer_like(&oty)
                    || matches!(oty, HirType::Slice(_))
                    || self.place_is_behind_borrow(object)
            }
            _ => false,
        }
    }

    pub fn check_move_receiver_not_borrowed(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        method_name: &str,
    ) {
        if self.in_unsafe() {
            return; // manual ownership is the programmer's job inside unsafe
        }
        let (value_ty, auto_deref) = match self.peek_type(object) {
            HirType::Ref { inner, .. }
            | HirType::SafePointer { inner, .. }
            | HirType::UnsafePointer { inner, .. } => (*inner, true),
            other => (other, false),
        };
        if self.copy_analysis.borrow().type_is_copy(&value_ty) {
            return;
        }
        if auto_deref || self.place_is_behind_borrow(object) {
            self.record(TypeErrorKind::Generic(format!(
                "cannot call `{}` on a value behind a reference: it takes `this` by value, \
                 which would move `{}` out of borrowed content",
                method_name,
                type_to_string(&value_ty)
            )));
        }
    }

    pub fn check_receiver_ref_kind(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        kind: &ThisPassingKind,
        method_name: &str,
    ) {
        let needed = match kind {
            ThisPassingKind::Move | ThisPassingKind::MoveMut => {
                self.check_move_receiver_not_borrowed(object, method_name);
                return;
            }
            ThisPassingKind::RefMut | ThisPassingKind::MutSafePtr => RefKind::Unique,
            ThisPassingKind::RefAlias => RefKind::Alias,
            _ => return,
        };
        let held = match self.peek_type(object) {
            HirType::Ref { ref_kind, .. } => ref_kind,
            _ => {
                let Some((root, _)) = self.static_field_path(object) else {
                    return;
                };
                let Some(&k) = self.local_ref_kind.get(&root) else {
                    return;
                };
                k
            }
        };
        let ok = match (held, needed) {
            (RefKind::Unique, _) => true,
            (RefKind::Alias, RefKind::Alias) => true,
            _ => false,
        };
        if !ok {
            let name = self
                .static_field_path(object)
                .map(|(r, _)| str_id_to_string(r))
                .unwrap_or_else(|| "this value".into());
            let (has, wants) = (
                match held {
                    RefKind::Shared => "`&`",
                    RefKind::Alias => "`&alias`",
                    RefKind::Unique => "`&mut`",
                },
                match needed {
                    RefKind::Alias => "`&alias`",
                    _ => "`&mut`",
                },
            );
            self.record(TypeErrorKind::Generic(format!(
                "cannot call `{}` on `{}`: it needs {} access but `{}` is only {}; \
                 a reference cannot be upgraded",
                method_name, name, wants, name, has
            )));
        }
    }

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
}

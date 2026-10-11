use ir::{
    ast::MutabilityState,
    borrow_checker::{
        BorrowKind, Bound, IndexContainer, IndexTemplate, Interval, MemoryRelation, PlaceId,
        ReadTemplate, TemplateBase, TemplateProjection,
    },
    errors::type_error::TypeErrorKind,
    hir::{
        AssignmentOperator, HirErrorHandlerPattern, HirExpr, HirFunc, HirParam, HirStmt, HirType,
        InterpolationPart, StrId,
    },
    ir_hasher::FxHashMap,
};

use crate::{TypeChecker, str_id_to_string};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// Cached read templates for `func`, one per normal parameter.
    /// An empty entry is inserted *before* building, so recursion (a function reading through a
    /// call to itself) terminates with "reads nothing yet" instead of looping.
    pub fn analyze_read_templates(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<ReadTemplate> {
        if let Some(t) = self.read_templates.get(&func.name) {
            return t.clone();
        }
        self.read_templates.insert(func.name, Vec::new());
        let templates = self.build_read_templates(func);
        self.read_templates.insert(func.name, templates.clone());
        templates
    }

    /// Walks the body, recording every read that goes through a parameter.
    /// Name resolution depends on the module, so `current_module_idx` is temporarily switched to
    /// the callee's declaring module and restored afterwards.
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

    /// Records one read.
    ///
    /// A read of the bare parameter (no projections) means the whole thing may be
    /// read, so the slot becomes `Opaque`; otherwise the path is appended (unless already `Opaque`).
    /// Reads rooted at `this` or a non-parameter are ignored.
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

    /// Handles the left side of `target = value`. A plain assignment *writes* the target, it
    /// doesn't read it, but reaching it reads the **prefix** (`a.b.c = v` reads `a.b`), so the last
    /// projection is dropped before recording. Targets that aren't parameter paths are walked as
    /// ordinary expressions.
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

    /// If `expr` is `inner as *mut T` (mutable raw/safe pointer cast), returns `inner`.
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

    /// Finds the function a call expression names, for plain functions and `module::function`
    /// paths (honouring per-module import aliases). Methods, interface calls and lambdas aren't
    /// resolved here (`None`).
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
    ///
    /// Used so that `callee(x as *mut T)` where the callee never reads `x` is treated as a *write*
    /// target rather than a read.
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

    /// An expression walker that first tries to read `expr` as a parameter path (`p`, `p.f`, `p[i]`, `*p`)
    /// and records it; otherwise it recurses into sub-expressions. Assignment targets and
    /// `callee(x as *mut T)` arguments are special-cased (see the write-target helper).
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
            HirExpr::Lambda { params, body, .. } => {
                // A captured parameter read inside the closure is still a read of that
                // parameter, so walk the body. Lambda params shadow outer ones, so hide
                // those names first (indices into `templates` stay valid since we only
                // remove map entries).
                if params.iter().any(|p| param_index.contains_key(&p.name)) {
                    let mut inner = param_index.clone();
                    for p in params.iter() {
                        inner.remove(&p.name);
                    }
                    self.collect_param_reads_stmt(body, &inner, has_this, templates);
                } else {
                    self.collect_param_reads_stmt(body, param_index, has_this, templates);
                }
            }
            // Ident/This/ModuleAccess/GenericIdent/literals/Undefined/
            // UnknownIntrinsic: either already handled by the
            // expr_to_template attempt above, or carry no sub-expressions.
            _ => {}
        }
    }

    /// Statement counterpart of [`Self::collect_param_reads_expr`]; just recurses structurally
    /// (including `let ... catch` handler bodies).
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

    /// Replays template projections onto a concrete caller-side place. `Index(Param(j))` is
    /// resolved from the j-th call argument; an unresolvable index stops the walk and returns the
    /// place reached so far (a *coarser*, safe over-approximation).
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

    /// True if the callee may read *through* the parameter (an index or deref), as opposed to only
    /// reading its own fields. `Opaque` counts as yes.
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

    /// For `f(&inner)`: errors if `f` may read a part of `inner` that is still covered by an active
    /// **mutable** loan on the same root. `Opaque` templates compare the whole argument place;
    /// `Paths` templates compare each read path (resolved onto the caller's place). Only a definite
    /// `Overlap` is reported.
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
                    let Some(loan) = self.borrow_checker.loan(*loan_id) else {
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
                        let Some(loan) = self.borrow_checker.loan(*loan_id) else {
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
}

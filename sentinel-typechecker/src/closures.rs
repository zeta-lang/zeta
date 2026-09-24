use super::*;
use ir::{
    borrow_checker::{BorrowKind, IndexContainer, Interval, LoanId},
    errors::type_error::TypeErrorKind,
    hir::{
        effect_path_is_prefix, effect_segment_eq, pattern_bound_names, static_effect_path,
        CaptureMode, ClosureKind, ClosureLowering, HirClosureCapture, HirEffectSegment,
        HirErrorHandlerPattern, HirExpr, HirFunc, HirGeneric, HirMatchArm, HirParam, HirStmt,
        HirType, InterpolationPart, RefKind, StrId,
    },
    ir_hasher::FxHashMap,
    span::SourceSpan,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum UseLevel {
    Read,
    Alias,
    Mut,
    Move,
}

impl From<BorrowKind> for UseLevel {
    fn from(k: BorrowKind) -> Self {
        match k {
            BorrowKind::Mutable => UseLevel::Mut,
            BorrowKind::Alias => UseLevel::Alias,
            BorrowKind::Shared => UseLevel::Read,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Usage {
    level: UseLevel,
    mutated: bool,
}

impl Default for Usage {
    fn default() -> Self {
        Usage {
            level: UseLevel::Read,
            mutated: false,
        }
    }
}

#[derive(Default)]
pub(super) struct ClosureFrame<'bump> {
    entries: Vec<(StrId, Vec<HirEffectSegment<'bump>>, Usage)>,
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub(super) fn note_capture_use(
        &self,
        root: StrId,
        path: &[HirEffectSegment<'bump>],
        level: UseLevel,
    ) {
        let mut frames = self.closure_frames.borrow_mut();
        for f in frames.iter_mut() {
            let hit: Option<&mut Usage> = f
                .entries
                .iter_mut()
                .find(|(r, p, _)| *r == root && effect_path_is_prefix(p, path))
                .map(|(_, _, usage)| usage);

            if let Some(usage) = hit {
                usage.level = usage.level.max(level);
                usage.mutated |= level == UseLevel::Mut;
            }

            // Cannot use or_else due to borrowck error :P love u NLL
            let other = f
                .entries
                .iter_mut()
                .find(|(r, p, _)| *r == root && p.is_empty())
                .map(|(_, _, usage)| usage);
            if let Some(usage) = other {
                usage.level = usage.level.max(level);
                usage.mutated |= level == UseLevel::Mut;
            }
        }
    }

    pub(super) fn mutation_path(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<HirEffectSegment<'bump>>)> {
        let through_ptr = |t: HirType<'a, 'bump>| {
            matches!(
                t,
                HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
            )
        };
        match expr {
            HirExpr::Slice { object, .. } => {
                if through_ptr(self.peek_type(object)) {
                    None
                } else {
                    self.mutation_path(object)
                }
            }
            HirExpr::Deref { expr: inner, .. } => {
                if through_ptr(self.peek_type(inner)) {
                    None
                } else {
                    self.mutation_path(inner)
                }
            }
            _ => self.effect_path_of(expr),
        }
    }

    pub(super) fn effect_path_of(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<HirEffectSegment<'bump>>)> {
        static_effect_path(expr, self.this_id, &self.context.bump)
    }

    fn is_enclosing_local(&self, n: StrId) -> bool {
        if n == self.this_id {
            self.context.get_variable("this").is_some()
        } else {
            self.context.is_local_binding(&str_id_to_string(n))
        }
    }

    fn expr_from_path(
        &self,
        root: StrId,
        path: &[HirEffectSegment<'bump>],
        span: SourceSpan<'a>,
    ) -> HirExpr<'a, 'bump> {
        let mut e = if root == self.this_id {
            HirExpr::This { span }
        } else {
            HirExpr::Ident(root, span)
        };
        for seg in path {
            e = match seg {
                HirEffectSegment::Field(f) => HirExpr::FieldAccess {
                    object: self.context.bump.alloc_value(e),
                    field: *f,
                    span,
                },
                HirEffectSegment::Index(_) => HirExpr::Index {
                    object: self.context.bump.alloc_value(e),
                    index: self.context.bump.alloc_value(HirExpr::Number(0, span)),
                    span,
                },
            };
        }
        e
    }

    fn capture_root_expr(&self, root: StrId, span: SourceSpan<'a>) -> HirExpr<'a, 'bump> {
        if root == self.this_id {
            HirExpr::This { span }
        } else {
            HirExpr::Ident(root, span)
        }
    }

    fn describe_path(&self, source: StrId, path: &[HirEffectSegment<'bump>]) -> String {
        let mut s = str_id_to_string(source);
        for seg in path {
            match seg {
                HirEffectSegment::Field(f) => {
                    s.push('.');
                    s.push_str(&str_id_to_string(*f));
                }
                HirEffectSegment::Index(_) => s.push_str("[..]"),
            }
        }
        s
    }

    fn fv_add(
        &self,
        root: StrId,
        path: Vec<HirEffectSegment<'bump>>,
        out: &mut Vec<(StrId, Vec<HirEffectSegment<'bump>>)>,
    ) {
        let dup = out.iter().any(|(r, p)| {
            *r == root
                && p.len() == path.len()
                && p.iter()
                    .zip(path.iter())
                    .all(|(a, b)| effect_segment_eq(a, b))
        });
        if !dup {
            out.push((root, path));
        }
    }

    fn fv_arm(
        &self,
        arm: &HirMatchArm<'a, 'bump>,
        bound: &mut Vec<StrId>,
        out: &mut Vec<(StrId, Vec<HirEffectSegment<'bump>>)>,
    ) {
        let m = bound.len();
        pattern_bound_names(&arm.pattern, bound);
        if let Some(g) = arm.guard {
            self.fv_expr(g, bound, out);
        }
        self.fv_stmt(arm.body, bound, out);
        bound.truncate(m);
    }

    fn fv_stmt(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        bound: &mut Vec<StrId>,
        out: &mut Vec<(StrId, Vec<HirEffectSegment<'bump>>)>,
    ) {
        match stmt {
            HirStmt::Let {
                name,
                value,
                else_block,
                catch_pattern,
                ..
            } => {
                self.fv_expr(value, bound, out);
                if let Some(b) = else_block {
                    self.fv_stmt(b, bound, out);
                }
                if let Some(pattern) = catch_pattern {
                    match pattern {
                        HirErrorHandlerPattern::Single { body, .. } => {
                            for s in body.iter() {
                                self.fv_stmt(s, bound, out);
                            }
                        }
                        HirErrorHandlerPattern::Multiple { branches } => {
                            for branch in branches.iter() {
                                for s in branch.body.iter() {
                                    self.fv_stmt(s, bound, out);
                                }
                            }
                        }
                    }
                }
                bound.push(*name);
            }
            HirStmt::Const(c) => {
                self.fv_expr(&c.value, bound, out);
                bound.push(c.name);
            }
            HirStmt::Return(Some(e), _) | HirStmt::Break(Some(e), _) => self.fv_expr(e, bound, out),
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => {}
            HirStmt::Expr(e) => self.fv_expr(e, bound, out),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.fv_expr(cond, bound, out);
                let m = bound.len();
                for s in then_block.iter() {
                    self.fv_stmt(s, bound, out);
                }
                bound.truncate(m);
                if let Some(e) = else_block {
                    self.fv_stmt(e, bound, out);
                }
            }
            HirStmt::While { cond, body } => {
                self.fv_expr(cond, bound, out);
                self.fv_stmt(body, bound, out);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                let m = bound.len();
                if let Some(i) = init {
                    self.fv_stmt(i, bound, out);
                }
                if let Some(c) = condition {
                    self.fv_expr(c, bound, out);
                }
                if let Some(inc) = increment {
                    self.fv_expr(inc, bound, out);
                }
                self.fv_stmt(body, bound, out);
                bound.truncate(m);
            }
            HirStmt::Block { body, span: _ } => {
                let m = bound.len();
                for s in body.iter() {
                    self.fv_stmt(s, bound, out);
                }
                bound.truncate(m);
            }
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.fv_expr(expr, bound, out);
                for arm in arms.iter() {
                    self.fv_arm(arm, bound, out);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => self.fv_stmt(body, bound, out),
        }
    }

    fn fv_expr(
        &self,
        expr: &HirExpr<'a, 'bump>,
        bound: &mut Vec<StrId>,
        out: &mut Vec<(StrId, Vec<HirEffectSegment<'bump>>)>,
    ) {
        match expr {
            HirExpr::Ident(n, _) => {
                if !bound.contains(n) {
                    self.fv_add(*n, Vec::new(), out);
                }
            }
            HirExpr::This { .. } => {
                if !bound.contains(&self.this_id) {
                    self.fv_add(self.this_id, Vec::new(), out);
                }
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                match self.effect_path_of(expr) {
                    Some((root, path)) if !bound.contains(&root) => self.fv_add(root, path, out),
                    Some(_) => {}
                    None => self.fv_expr(object, bound, out),
                }
            }
            HirExpr::Index { object, index, .. } => {
                match self.effect_path_of(expr) {
                    Some((root, path)) if !bound.contains(&root) => self.fv_add(root, path, out),
                    Some(_) => {}
                    None => self.fv_expr(object, bound, out),
                }
                self.fv_expr(index, bound, out);
            }
            HirExpr::Assignment { target, value, .. } => {
                self.fv_expr(target, bound, out);
                self.fv_expr(value, bound, out);
            }
            HirExpr::Ref { expr: inner, .. }
            | HirExpr::Deref { expr: inner, .. }
            | HirExpr::Cast { expr: inner, .. } => self.fv_expr(inner, bound, out),
            HirExpr::Slice {
                object, start, end, ..
            } => {
                self.fv_expr(object, bound, out);
                self.fv_expr(start, bound, out);
                self.fv_expr(end, bound, out);
            }
            HirExpr::Range { start, end, .. } => {
                self.fv_expr(start, bound, out);
                self.fv_expr(end, bound, out);
            }
            HirExpr::Tuple(exprs, _)
            | HirExpr::ArrayLiteral {
                elements: exprs, ..
            }
            | HirExpr::ExprList { list: exprs, .. } => {
                for e in exprs.iter() {
                    self.fv_expr(e, bound, out);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.fv_expr(left, bound, out);
                self.fv_expr(right, bound, out);
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                self.fv_expr(callee, bound, out);
                for a in args.iter() {
                    self.fv_expr(a, bound, out);
                }
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.fv_expr(&f.value, bound, out);
                }
            }
            HirExpr::EnumInit { args, .. } | HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.fv_expr(a, bound, out);
                }
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.fv_expr(e, bound, out);
                    }
                }
            }
            HirExpr::If { if_stmt, .. } => self.fv_stmt(if_stmt, bound, out),
            HirExpr::Match {
                expr: scrutinee,
                arms,
                ..
            } => {
                self.fv_expr(scrutinee, bound, out);
                for arm in arms.iter() {
                    self.fv_arm(arm, bound, out);
                }
            }
            HirExpr::Block { body, .. } => {
                let m = bound.len();
                for s in body.iter() {
                    self.fv_stmt(s, bound, out);
                }
                bound.truncate(m);
            }
            HirExpr::Lambda { params, body, .. } => {
                let m = bound.len();
                bound.extend(params.iter().map(|p| p.name));
                self.fv_stmt(body, bound, out);
                bound.truncate(m);
            }
            HirExpr::Null(_)
            | HirExpr::Number(..)
            | HirExpr::Char(..)
            | HirExpr::String(..)
            | HirExpr::Boolean(..)
            | HirExpr::Decimal(..)
            | HirExpr::Undefined { .. }
            | HirExpr::Uninit { .. }
            | HirExpr::GenericIdent(..)
            | HirExpr::ModuleAccess(_)
            | HirExpr::UnknownIntrinsic { .. } => {}
        }
    }

    fn coalesce_free_paths(
        mut entries: Vec<(StrId, Vec<HirEffectSegment<'bump>>)>,
    ) -> Vec<(StrId, Vec<HirEffectSegment<'bump>>)> {
        entries.sort_by_key(|(_, p)| p.len());
        let mut kept: Vec<(StrId, Vec<HirEffectSegment<'bump>>)> = Vec::new();
        'outer: for (root, path) in entries {
            for (kr, kp) in &kept {
                if *kr == root && effect_path_is_prefix(kp, &path) {
                    continue 'outer;
                }
            }
            kept.push((root, path));
        }
        kept
    }

    fn finalize_captures(
        &self,
        frame: &ClosureFrame<'bump>,
        span: SourceSpan<'a>,
    ) -> (
        Vec<HirClosureCapture<'bump>>,
        Vec<HirType<'a, 'bump>>,
        ClosureKind,
    ) {
        let mut captures = Vec::with_capacity(frame.entries.len());
        let mut field_tys = Vec::with_capacity(frame.entries.len());
        let (mut any_move, mut any_mut) = (false, false);

        for (i, (root, path, usage)) in frame.entries.iter().enumerate() {
            let mode = match usage.level {
                UseLevel::Read => CaptureMode::ByRef(RefKind::Shared),
                UseLevel::Alias => CaptureMode::ByRef(RefKind::Alias),
                UseLevel::Mut => CaptureMode::ByRef(RefKind::Unique),
                UseLevel::Move => CaptureMode::ByValue,
            };
            any_move |= usage.level == UseLevel::Move;
            any_mut |= usage.mutated;

            let place_expr = self.expr_from_path(*root, path, span);
            let place_ty = self.peek_type(&place_expr);
            let field_ty = match mode {
                CaptureMode::ByValue => place_ty,
                CaptureMode::ByRef(rk) => HirType::Ref {
                    inner: self.context.bump.alloc_value(place_ty),
                    ref_kind: rk,
                    provenance: self.infer_provenance(&place_expr),
                },
            };

            let field_name = StrId(self.context.string_pool.intern(&format!("__cap_{}", i)));
            captures.push(HirClosureCapture {
                name: field_name,
                mode,
                source: *root,
                source_path: self.context.bump.alloc_slice(path),
            });
            field_tys.push(field_ty);
        }

        let kind = if any_move {
            ClosureKind::FnOnce
        } else if any_mut {
            ClosureKind::FnMut
        } else {
            ClosureKind::Fn
        };
        (captures, field_tys, kind)
    }

    fn register_capture_loans(
        &mut self,
        captures: &[HirClosureCapture<'bump>],
        span: SourceSpan<'a>,
    ) -> Vec<LoanId> {
        let mut loans = Vec::new();
        for c in captures {
            let CaptureMode::ByRef(rk) = c.mode else {
                continue;
            };
            let root_expr = self.capture_root_expr(c.source, span);
            let Some(mut place) = self.resolve_place(&root_expr) else {
                continue;
            };
            for seg in c.source_path {
                place = match seg {
                    HirEffectSegment::Field(f) => self.borrow_checker.project_field(place, *f),
                    HirEffectSegment::Index(key) => {
                        let bound = self.effect_index_key_to_bound(key);
                        let interval = Interval {
                            lower: bound.clone(),
                            upper: bound,
                        };
                        self.borrow_checker.project_index(
                            place,
                            interval,
                            IndexContainer::Primitive,
                        )
                    }
                };
            }
            let result = match rk {
                RefKind::Unique => self.borrow_checker.borrow_mut(place),
                RefKind::Alias => self.borrow_checker.borrow_alias(place),
                RefKind::Shared => self.borrow_checker.borrow_shared(place),
            };
            match result {
                Ok(id) => loans.push(id),
                Err(e) => {
                    let msg = self.describe_borrow_error(&e, None);
                    self.record(TypeErrorKind::Generic(msg));
                }
            }
        }
        loans
    }

    pub(super) fn check_lambda(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        sig: Option<HirType<'a, 'bump>>,
        as_closure: bool,
    ) -> HirType<'a, 'bump> {
        let HirExpr::Lambda {
            params,
            return_type,
            body,
            span,
            ..
        } = expr
        else {
            unreachable!("check_lambda on a non-lambda");
        };
        self.set_span(*span);
        let declared_ret: HirType<'a, 'bump> = **return_type;
        let ret_is_inferred = matches!(declared_ret, HirType::Unknown | HirType::Void);

        let mut bound: Vec<StrId> = params.iter().map(|p| p.name).collect();
        let mut raw_free = Vec::new();
        self.fv_stmt(body, &mut bound, &mut raw_free);
        let free = Self::coalesce_free_paths(
            raw_free
                .into_iter()
                .filter(|(root, _)| self.is_enclosing_local(*root))
                .collect(),
        );

        let (sig_params, sig_ret): (Option<&[HirType<'a, 'bump>]>, Option<HirType<'a, 'bump>>) =
            match &sig {
                Some(HirType::Lambda {
                    params: sp,
                    return_type: sr,
                }) => (Some(*sp), Some(**sr)),
                _ => (None, None),
            };
        let mut param_tys: Vec<HirType<'a, 'bump>> = Vec::with_capacity(params.len());
        for (i, p) in params.iter().enumerate() {
            let from_sig = sig_params
                .and_then(|sp| sp.get(i))
                .copied()
                .filter(|t| !matches!(t, HirType::Generic(_)));
            let ty = match (p.param_type, from_sig) {
                (Some(annotated), Some(expected)) => {
                    self.recover(self.types_compatible(&expected, &annotated), ());
                    annotated
                }
                (Some(annotated), None) => annotated,
                (None, Some(expected)) => expected,
                (None, None) => {
                    if as_closure {
                        self.record(TypeErrorKind::TypeCannotBeInferred);
                    }
                    HirType::Unknown
                }
            };
            param_tys.push(ty);
        }

        let mut lambda_context = self.context.create_child_scope();
        for (p, ty) in params.iter().zip(param_tys.iter()) {
            let param_name = str_id_to_string(p.name);
            if lambda_context.get_variable(&param_name).is_some() {
                self.record(TypeErrorKind::VariableAlreadyExists {
                    var_name: param_name.clone(),
                });
            }
            let sid = self.mint_symbol_id();
            lambda_context.add_variable(param_name, *ty, sid);
        }

        lambda_context.current_return_type = if ret_is_inferred {
            None
        } else {
            Some(declared_ret)
        };
        lambda_context.in_loop = false;

        self.closure_frames.borrow_mut().push(ClosureFrame {
            entries: free
                .iter()
                .map(|(r, p)| (*r, p.clone(), Usage::default()))
                .collect(),
        });
        let old_context = std::mem::replace(&mut self.context, lambda_context);
        let body_ty = self.check_stmt(body);
        self.context = old_context;
        let frame = self.closure_frames.borrow_mut().pop().unwrap_or_default();

        let concrete_sig_ret = sig_ret.filter(|t| !matches!(t, HirType::Generic(_)));
        let ret_ty = if ret_is_inferred {
            match (concrete_sig_ret, body_ty) {
                (Some(sr), Some(t)) if !matches!(t, HirType::Never) && sr != HirType::Void => {
                    self.recover(self.types_compatible(&sr, &t), ());
                    sr
                }
                (Some(sr), _) => sr,
                (None, Some(t)) if !matches!(t, HirType::Never) => t,
                (None, _) => HirType::Void,
            }
        } else {
            if let Some(t) = body_ty {
                self.recover(self.types_compatible(&declared_ret, &t), ());
            }
            declared_ret
        };

        if let Some(HirType::Lambda {
            params: sp,
            return_type: sr,
        }) = &sig
        {
            let mut subs = FxHashMap::default();
            for (d, a) in sp.iter().zip(param_tys.iter()) {
                self.unify_generic(d, a, &mut subs);
            }
            self.unify_generic(sr, &ret_ty, &mut subs);
            for (k, v) in subs {
                self.closure_generic_subs.entry(k).or_insert(v);
            }
            if !matches!(**sr, HirType::Generic(_)) {
                self.recover(self.types_compatible(sr, &ret_ty), ());
            }
        }

        let (captures, field_tys, kind) = self.finalize_captures(&frame, *span);

        if !as_closure {
            for c in &captures {
                let desc = self.describe_path(c.source, c.source_path);
                self.record(TypeErrorKind::Generic(format!(
                    "a function-pointer lambda cannot capture `{}`; take a generic parameter \
                     constrained by `func(..)` to use a closure",
                    desc
                )));
            }
            return HirType::Lambda {
                params: self.context.bump.alloc_slice(&param_tys),
                return_type: self.context.bump.alloc_value(ret_ty),
            };
        }

        let key = Self::stmt_key(body);
        let (env_name, fn_name) = match self.closure_table.get(&key) {
            Some(prev) => (prev.env_name, prev.fn_name),
            None => {
                let id = self.next_closure_id;
                self.next_closure_id += 1;
                (
                    StrId(
                        self.context
                            .string_pool
                            .intern(&format!("__closure_env_{}", id)),
                    ),
                    StrId(
                        self.context
                            .string_pool
                            .intern(&format!("__closure_fn_{}", id)),
                    ),
                )
            }
        };
        let env_ty = HirType::Struct {
            name: env_name,
            field_types: self.context.bump.alloc_slice(&field_tys),
            type_args: &[],
        };

        for c in &captures {
            let lvl = match c.mode {
                CaptureMode::ByValue => UseLevel::Move,
                CaptureMode::ByRef(RefKind::Unique) => UseLevel::Mut,
                CaptureMode::ByRef(RefKind::Alias) => UseLevel::Alias,
                CaptureMode::ByRef(RefKind::Shared) => UseLevel::Read,
            };
            self.note_capture_use(c.source, c.source_path, lvl);
        }

        let loans = self.register_capture_loans(&captures, *span);
        self.closure_loans.insert(Self::expr_key(expr), loans);
        self.closure_table.insert(
            key,
            ClosureLowering {
                env_name,
                fn_name,
                env_ty,
                captures,
                param_tys,
                ret_ty,
                kind,
            },
        );
        env_ty
    }

    fn is_lambda_constraint(c: &HirType<'a, 'bump>) -> bool {
        matches!(c, HirType::Lambda { .. })
    }

    pub(super) fn has_closure_generics(gs: &[HirGeneric<'a, 'bump>]) -> bool {
        gs.iter()
            .any(|g| g.constraints.iter().any(|c| Self::is_lambda_constraint(c)))
    }

    fn closure_constraint_for(
        &self,
        pt: &HirType<'a, 'bump>,
        callee: Option<&HirFunc<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        let HirType::Generic(g) = pt else {
            return None;
        };
        let generics = callee?.generics?;
        let hg = generics.iter().find(|h| h.name == *g)?;
        let c = hg
            .constraints
            .iter()
            .find(|c| Self::is_lambda_constraint(c))?;
        Some(self.substitute_type_local(c, &self.closure_pre_subs))
    }

    pub(super) fn pre_infer_generics(
        &self,
        callee: Option<&HirFunc<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
        params: &[HirParam<'a, 'bump>],
    ) -> FxHashMap<StrId, HirType<'a, 'bump>> {
        let mut subs = FxHashMap::default();
        let Some(gs) = callee.and_then(|f| f.generics) else {
            return subs;
        };
        if !Self::has_closure_generics(gs) {
            return subs;
        }
        for (arg, param) in args.iter().zip(params.iter()) {
            if matches!(arg, HirExpr::Lambda { .. }) {
                continue;
            }
            let Some(pt) = param.get_type() else { continue };
            let e = match arg {
                HirExpr::Ref { expr, .. } => &**expr,
                other => other,
            };
            let at = self.peek_type(e);
            if !matches!(at, HirType::Unknown) {
                self.unify_generic(pt, &at, &mut subs);
            }
        }
        subs
    }

    pub(super) fn check_arg_expr(
        &mut self,
        arg: &HirExpr<'a, 'bump>,
        param_type: Option<&HirType<'a, 'bump>>,
        callee: Option<&HirFunc<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        match param_type {
            Some(pt) => {
                if matches!(arg, HirExpr::Lambda { .. }) {
                    if let Some(c) = self.closure_constraint_for(pt, callee) {
                        return self.check_lambda(arg, Some(c), true);
                    }
                }
                self.check_expr_expected(arg, pt)
            }
            None => self.check_expr(arg),
        }
    }

    pub(super) fn check_closure_call(
        &mut self,
        generic: StrId,
        args: &[HirExpr<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        let Some(HirType::Lambda {
            params,
            return_type,
        }) = self.fn_closure_constraints.get(&generic).copied()
        else {
            return HirType::Unknown;
        };
        if args.len() != params.len() {
            self.record(TypeErrorKind::InvalidFunctionCall {
                expected_args: params.len(),
                found_args: args.len(),
            });
        }
        for (arg, pt) in args.iter().zip(params.iter()) {
            let at = self.check_expr_expected(arg, pt);
            self.check_and_record_value_use(arg, &at);
            self.recover(self.types_compatible(pt, &at), ());
        }
        *return_type
    }

    pub fn closure_table(&self) -> &FxHashMap<usize, ClosureLowering<'a, 'bump>> {
        &self.closure_table
    }
}

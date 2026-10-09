use ir::{
    borrow_checker::LoanId,
    errors::type_error::TypeErrorKind,
    hir::{CaptureMode, HirExpr, HirFunc, HirParam, HirType, ProvenanceRoot, StrId},
    nll_cfg::PointId,
};

use crate::{
    TypeChecker,
    auto_traits::AutoTrait,
    borrow_lifetime::{CalleeConcurrency, SuspensionReport},
    naming::type_to_string,
    str_id_to_string,
};

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// Call for every function and method when its module is registered.
    pub fn register_callee_concurrency(&mut self, func: &HirFunc<'a, 'bump>) {
        let (Some(generics), Some(params)) = (func.generics, func.params) else {
            return;
        };

        let mut normal: Vec<(usize, HirType<'a, 'bump>)> = Vec::new();
        let mut idx = 0usize;
        for p in params.iter() {
            if let HirParam::Normal { param_type, .. } = p {
                normal.push((idx, *param_type));
                idx += 1;
            }
        }

        let mut cc = CalleeConcurrency::default();
        for g in generics.iter() {
            let is_static = g
                .min_provenance
                .is_some_and(|m| matches!(m.root, ProvenanceRoot::Static));
            if is_static {
                for (i, pty) in &normal {
                    if matches!(pty, HirType::Generic(n) if *n == g.name) {
                        cc.static_params.push(*i);
                    }
                }
            }
            let traits: Vec<AutoTrait> = g
                .constraints
                .iter()
                .filter_map(|c| {
                    let display = type_to_string(c);
                    AutoTrait::from_interface_name(&display)
                })
                .collect();
            if traits.is_empty() {
                continue;
            }

            for (i, pty) in &normal {
                if matches!(pty, HirType::Generic(n) if *n == g.name) {
                    cc.auto_bounds.push((*i, traits.clone()));
                }
            }
        }

        if !cc.auto_bounds.is_empty() || !cc.static_params.is_empty() {
            self.callee_concurrency.insert(func.name, cc);
        }
    }

    /// Call at every call site that resolved a `HirFunc`, after the arguments were checked
    /// and any closure loans were created, and before temporary loans are ended.
    pub fn apply_callee_concurrency(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
    ) {
        let Some(cc) = self.callee_concurrency.get(&func.name).cloned() else {
            return;
        };
        for (idx, traits) in &cc.auto_bounds {
            if let Some(arg) = args.get(*idx) {
                self.check_arg_auto_bounds(arg, traits);
            }
        }
        for idx in &cc.static_params {
            if let Some(arg) = args.get(*idx) {
                self.check_arg_static_bound(func.name, arg);
            }
        }
    }

    /// `T: &static`; the argument may not hold a borrow of anything that is not `&static`.
    /// A closure that captures a local by reference holds exactly such a borrow.
    fn check_arg_static_bound(&mut self, callee: StrId, arg: &HirExpr<'a, 'bump>) {
        let callee_name = str_id_to_string(callee);
        let mut problems: Vec<String> = Vec::new();

        let (captures, ret_ty) = match arg {
            HirExpr::Lambda { body, .. } => match self.closure_table.get(&Self::stmt_key(body)) {
                Some(l) => (Some(l.captures.clone()), Some(l.ret_ty)),
                None => (None, None),
            },
            _ => (None, None),
        };

        if let Some(rt) = ret_ty {
            if Self::type_holds_nonstatic_ref(&rt) {
                problems.push(format!(
                    "the closure returns `{}`, which contains a non-`&static` reference; \
                     copy the data out before the closure ends",
                    type_to_string(&rt)
                ));
            }
        }

        if let (HirExpr::Lambda { span, .. }, Some(caps)) = (arg, captures) {
            for c in caps.iter() {
                let who = str_id_to_string(c.source);
                match c.mode {
                    CaptureMode::ByRef(kind) => problems.push(format!(
                        "the closure borrows `{who}` ({kind:?}), which is not `&static`; \
                         use `move` and share it through an `Arc` (or use `thread::scope`)"
                    )),
                    CaptureMode::ByValue => {
                        let place = self.expr_from_path(c.source, c.source_path, *span);
                        let ty = self.peek_type(&place);
                        if Self::type_holds_nonstatic_ref(&ty) {
                            problems.push(format!(
                                "the closure moves `{who}` of type `{}`, which contains a \
                                 non-`&static` reference",
                                type_to_string(&ty)
                            ));
                        }
                    }
                }
            }
        } else {
            let ty = self.peek_type(arg);
            if Self::type_holds_nonstatic_ref(&ty) {
                problems.push(format!(
                    "argument of type `{}` contains a non-`&static` reference",
                    type_to_string(&ty)
                ));
            }
        }

        for p in problems {
            self.record(TypeErrorKind::Generic(format!(
                "`{callee_name}` requires `&static` here, but {p}"
            )));
        }
    }

    fn type_holds_nonstatic_ref(ty: &HirType<'a, 'bump>) -> bool {
        match ty {
            HirType::Ref {
                inner, provenance, ..
            } => {
                !provenance.is_some_and(|p| matches!(p.root, ProvenanceRoot::Static))
                    || Self::type_holds_nonstatic_ref(inner)
            }
            HirType::Struct {
                field_types,
                type_args,
                ..
            } => field_types
                .iter()
                .chain(type_args.iter())
                .any(Self::type_holds_nonstatic_ref),
            HirType::Enum { type_args, .. } => type_args.iter().any(Self::type_holds_nonstatic_ref),
            HirType::SafePointer { inner, .. }
            | HirType::UnsafePointer { inner, .. }
            | HirType::Nullable(inner)
            | HirType::Array(inner, _)
            | HirType::Slice(inner) => Self::type_holds_nonstatic_ref(inner),
            HirType::OwnedPointer { inner, .. } => Self::type_holds_nonstatic_ref(inner),
            HirType::Tuple(ts) => ts.iter().any(Self::type_holds_nonstatic_ref),
            _ => false,
        }
    }

    /// Call at the start of `check_function`.
    pub fn begin_concurrency_function(&mut self, func: &HirFunc<'a, 'bump>) {
        self.current_fn_bounds.clear();
        if let Some(gs) = func.generics {
            for g in gs.iter() {
                let bounds: Vec<String> = g.constraints.iter().map(|c| type_to_string(c)).collect();
                self.current_fn_bounds.insert(g.name, bounds);
            }
        }
    }

    pub fn analyze_suspension(&self, point: PointId) -> SuspensionReport {
        let mut frame_locals: Vec<StrId> = self
            .init_state
            .keys()
            .copied()
            .filter(|l| self.local_used_after(point, *l))
            .collect();
        frame_locals.sort_by_key(|l| str_id_to_string(*l));

        let borrows_across: Vec<LoanId> = self
            .loan_owners
            .iter()
            .filter(|(_, owner)| self.local_used_after(point, **owner))
            .map(|(l, _)| *l)
            .collect();

        SuspensionReport {
            frame_locals,
            borrows_across,
        }
    }

    /// Everything live across a suspension becomes part of the async frame. When the
    /// computation may run on another thread the frame has to be `Send`.
    pub fn check_suspension_point(
        &mut self,
        point: PointId,
        require_send: bool,
    ) -> SuspensionReport {
        let report = self.analyze_suspension(point);
        if require_send {
            for local in report.frame_locals.iter() {
                let Some((_, ty)) = self.context.get_variable(&str_id_to_string(*local)) else {
                    continue;
                };
                if !self.implements_auto(&ty, AutoTrait::Send) {
                    self.record(TypeErrorKind::Generic(format!(
                        "async computation cannot be sent between threads: `{}` of type `{}` \
                         is held across a suspension point and is not `Send`",
                        str_id_to_string(*local),
                        type_to_string(&ty)
                    )));
                }
            }
        }
        report
    }
}

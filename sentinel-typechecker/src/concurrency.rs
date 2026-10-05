use ir::{
    borrow_checker::LoanId,
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, HirFunc, HirParam, HirType, StrId, ThisPassingKind},
    nll_cfg::PointId,
};

use crate::{
    auto_traits::AutoTrait,
    borrow_lifetime::{
        CalleeConcurrency, Holder, ObState, ObligationArm, ObligationBranch, SuspensionReport,
    },
    naming::type_to_string,
    str_id_to_string, TypeChecker,
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
                    if !traits.is_empty() {
                        cc.auto_bounds.push((*i, traits.clone()));
                    }
                }
            }
        }

        if !cc.auto_bounds.is_empty() || !cc.borrow_until.is_empty() {
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
        self.begin_borrow_until(func.name, &cc.borrow_until, args);
    }

    pub fn on_method_call(
        &mut self,
        receiver: &HirExpr<'a, 'bump>,
        method: &str,
        this_kind: &ThisPassingKind,
    ) {
        let consumes = matches!(this_kind, ThisPassingKind::Move | ThisPassingKind::MoveMut);
        let recv = match receiver {
            HirExpr::Ident(n, _) => Some(*n),
            _ => None,
        };

        let hits: Vec<usize> = self
            .borrow_obligations
            .iter()
            .enumerate()
            .filter(|(_, ob)| {
                if ob.discharged {
                    return false;
                }
                let by_method = ob.methods.iter().any(|m| m.as_str() == method);
                let by_move = ob.on_move && consumes;
                if !(by_method || by_move) {
                    return false;
                }
                match (ob.holder, recv) {
                    (Holder::Local(h), Some(n)) => h == n,
                    (Holder::InFlight(h), Some(n)) => h == n && consumes,
                    (Holder::Pending, None) => true, // spawn(..).join()
                    _ => false,
                }
            })
            .map(|(i, _)| i)
            .collect();

        for i in hits {
            self.discharge(i);
        }
    }

    /// Call at the start of `check_function`.
    pub fn begin_concurrency_function(&mut self, func: &HirFunc<'a, 'bump>) {
        self.borrow_obligations.clear();
        self.pinned_loans.clear();
        self.obligation_branches.clear();
        self.current_fn_bounds.clear();
        if let Some(gs) = func.generics {
            for g in gs.iter() {
                let bounds: Vec<String> = g.constraints.iter().map(|c| type_to_string(c)).collect();
                self.current_fn_bounds.insert(g.name, bounds);
            }
        }
    }

    fn snapshot_obligation_states(&self) -> Vec<ObState> {
        self.borrow_obligations
            .iter()
            .map(|o| ObState {
                discharged: o.discharged,
                holder: o.holder,
            })
            .collect()
    }

    pub fn begin_obligation_branch(&mut self) -> ObligationBranch {
        let base_len = self.borrow_obligations.len();
        self.obligation_branches.push(base_len);
        ObligationBranch {
            base_len,
            base: self.snapshot_obligation_states(),
        }
    }

    /// Closes one arm. Obligations created inside the arm die with it, so anything still
    /// outstanding there is a dropped handle.
    pub fn end_obligation_arm(&mut self, br: &ObligationBranch) -> ObligationArm {
        let local: Vec<usize> = (br.base_len..self.borrow_obligations.len())
            .filter(|&i| !self.borrow_obligations[i].discharged)
            .collect();
        for i in local {
            let msg = self.outstanding_message(i, "at the end of this branch");
            self.record(TypeErrorKind::Generic(msg));
        }

        let tail = self.borrow_obligations.split_off(br.base_len);
        for ob in tail {
            for l in ob.loans {
                self.pinned_loans.remove(&l);
                self.borrow_checker.end_loan_now(l);
            }
        }

        ObligationArm {
            states: self.snapshot_obligation_states(),
        }
    }

    /// Resets the path state to what it was when the branch started (for the next arm).
    pub fn restore_obligations(&mut self, br: &ObligationBranch) {
        for (i, st) in br.base.iter().enumerate() {
            if let Some(ob) = self.borrow_obligations.get_mut(i) {
                ob.discharged = st.discharged;
                ob.holder = st.holder;
            }
        }
    }

    /// Joins the arms that can fall through. An obligation is discharged afterwards only if
    /// it is discharged on all of them, and must have the same holder on the ones where it
    /// is still outstanding.
    pub fn join_obligation_arms(&mut self, br: ObligationBranch, arms: Vec<(ObligationArm, bool)>) {
        self.obligation_branches.pop();
        if arms.is_empty() {
            return;
        }

        let mut live: Vec<&ObligationArm> =
            arms.iter().filter(|(_, d)| !*d).map(|(a, _)| a).collect();
        if live.is_empty() {
            // Every arm diverges: the code after the join is unreachable.
            live = arms.iter().map(|(a, _)| a).collect();
        }

        let mut conflicted: Vec<usize> = Vec::new();
        for i in 0..br.base_len {
            let states: Vec<ObState> = live.iter().map(|a| a.states[i]).collect();
            let discharged = states.iter().all(|s| s.discharged);

            let (holder, conflict) = if discharged {
                (states[0].holder, false)
            } else {
                let outstanding: Vec<Holder> = states
                    .iter()
                    .filter(|s| !s.discharged)
                    .map(|s| s.holder)
                    .collect();
                let first = outstanding[0];
                (first, outstanding.iter().any(|h| *h != first))
            };

            let ob = &mut self.borrow_obligations[i];
            ob.discharged = discharged;
            ob.holder = holder;
            if conflict {
                conflicted.push(i);
            }
        }

        for i in conflicted {
            let callee = str_id_to_string(self.borrow_obligations[i].callee);
            self.record(TypeErrorKind::Generic(format!(
                "the handle from `{callee}` is held by different variables (or moved) on different \
                 branches, so its `borrow_until` obligation can't be tracked past this join; keep \
                 it in one variable on every branch"
            )));
            self.borrow_obligations[i].discharged = true;
        }

        self.flush_deferred_releases();
    }

    /// Loops: the body may run zero times, so the state after the loop is the join of the
    /// state before it and the state after one pass through the body.
    pub fn join_obligation_loop(&mut self, br: ObligationBranch, body: ObligationArm) {
        self.restore_obligations(&br);
        let skipped = self.end_obligation_arm(&br);
        self.join_obligation_arms(br, vec![(body, false), (skipped, false)]);
    }

    pub fn analyze_suspension(&self, point: PointId) -> SuspensionReport {
        let mut frame_locals: Vec<StrId> = self
            .init_state
            .keys()
            .copied()
            .filter(|l| self.local_used_after(point, *l))
            .collect();
        frame_locals.sort_by_key(|l| str_id_to_string(*l));

        let mut borrows_across: Vec<LoanId> = self
            .loan_owners
            .iter()
            .filter(|(_, owner)| self.local_used_after(point, **owner))
            .map(|(l, _)| *l)
            .collect();
        for l in self.pinned_loans.iter() {
            if !borrows_across.contains(l) {
                borrows_across.push(*l);
            }
        }

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

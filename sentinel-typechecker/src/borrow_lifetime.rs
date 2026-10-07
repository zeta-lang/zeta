use ir::{
    borrow_checker::LoanId,
    errors::type_error::TypeErrorKind,
    hir::{HirExpr, StrId},
};

use crate::{TypeChecker, auto_traits::AutoTrait, str_id_to_string};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Holder {
    /// Call result not bound yet.
    Pending,
    /// Held by a local; `join` etc. on it discharges.
    Local(StrId),
    /// Moved out of a local, waiting to be re-bound by a `let`.
    InFlight(StrId),
}

#[derive(Clone, Debug)]
pub struct BorrowObligation {
    pub callee: StrId,
    pub loans: Vec<LoanId>,
    pub holder: Holder,
    pub methods: Vec<StrId>,
    pub on_move: bool,
    /// Discharged on the *current path*. The loans may still be held back (see `discharge`).
    pub discharged: bool,
}

/// Path-dependent part of an obligation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ObState {
    pub discharged: bool,
    pub holder: Holder,
}

/// Handle returned by `begin_obligation_branch`.
pub struct ObligationBranch {
    pub(crate) base_len: usize,
    pub(crate) base: Vec<ObState>,
}

/// Obligation state at the end of one arm of a branch.
pub struct ObligationArm {
    pub(crate) states: Vec<ObState>,
}

#[derive(Clone, Debug, Default)]
pub struct CalleeConcurrency {
    /// (normal param index, Send/Sync bounds on that param's generic type)
    pub auto_bounds: Vec<(usize, Vec<AutoTrait>)>,
    /// Normal param indices whose generic is declared `&static`.
    pub static_params: Vec<usize>,
}

pub struct SuspensionReport {
    /// Locals that live in the async frame across the suspension.
    pub frame_locals: Vec<StrId>,
    /// Loans that survive the suspension.
    pub borrows_across: Vec<LoanId>,
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    /// Takes the loans that the argument holds out of the "ends with the call" bookkeeping.
    fn detach_arg_loans(&mut self, arg: &HirExpr<'a, 'bump>) -> Vec<LoanId> {
        match arg {
            HirExpr::Lambda { .. } => self
                .closure_loans
                .remove(&Self::expr_key(arg))
                .unwrap_or_default(),
            HirExpr::Ident(name, _) => {
                let owned: Vec<LoanId> = self
                    .loan_owners
                    .iter()
                    .filter(|(_, o)| **o == *name)
                    .map(|(l, _)| *l)
                    .collect();
                for l in &owned {
                    self.loan_owners.remove(l);
                }
                owned
            }
            _ => Vec::new(),
        }
    }

    /// NLL expiry must skip pinned loans: they end by discharge, never by liveness.
    pub fn is_loan_pinned(&self, id: LoanId) -> bool {
        self.pinned_loans.contains(&id)
    }

    /// `let holder = <expr>`: obligations produced by `<expr>` (or moved out of another
    /// local) now belong to `holder`.
    pub fn bind_obligations_to(&mut self, holder: StrId) {
        for ob in self.borrow_obligations.iter_mut() {
            if ob.discharged {
                continue;
            }
            if matches!(ob.holder, Holder::Pending | Holder::InFlight(_)) {
                ob.holder = Holder::Local(holder);
            }
        }
    }

    /// Call from `record_move` for whole-value moves of a non-Copy local.
    pub fn note_obligation_holder_moved(&mut self, name: StrId) {
        for ob in self.borrow_obligations.iter_mut() {
            if !ob.discharged && ob.holder == Holder::Local(name) {
                ob.holder = Holder::InFlight(name);
            }
        }
    }

    /// An obligation created before the innermost open branch is shared by all of its arms.
    pub fn obligation_is_branch_shared(&self, idx: usize) -> bool {
        self.obligation_branches
            .last()
            .is_some_and(|&base| idx < base)
    }

    /// Marks the obligation discharged *on the current path*. Its loans are only ended when
    /// no sibling arm still needs them; otherwise they are held until the branches join
    /// (`end_loan_now` can't be undone, so ending them in one arm would unprotect the other).
    pub fn discharge(&mut self, idx: usize) {
        self.borrow_obligations[idx].discharged = true;
        if !self.obligation_is_branch_shared(idx) {
            self.release_loans(idx);
        }
    }

    pub fn release_loans(&mut self, idx: usize) {
        let loans = std::mem::take(&mut self.borrow_obligations[idx].loans);
        for l in loans {
            self.pinned_loans.remove(&l);
            self.borrow_checker.end_loan_now(l);
        }
    }

    /// Ends the loans of every discharged obligation that is no longer shared.
    pub fn flush_deferred_releases(&mut self) {
        let ready: Vec<usize> = (0..self.borrow_obligations.len())
            .filter(|&i| {
                let ob = &self.borrow_obligations[i];
                ob.discharged && !ob.loans.is_empty() && !self.obligation_is_branch_shared(i)
            })
            .collect();
        for i in ready {
            self.release_loans(i);
        }
    }

    pub fn outstanding_message(&self, idx: usize, scope: &str) -> String {
        let ob = &self.borrow_obligations[idx];
        let callee = str_id_to_string(ob.callee);
        match ob.holder {
            Holder::Local(h) => format!(
                "`{}` goes out of scope {scope} with an outstanding `borrow_until` obligation \
                     from `{callee}`; call one of its discharging operations first, \
                     because dropping the handle does not stop the computation",
                str_id_to_string(h)
            ),
            Holder::Pending => format!(
                "the value returned by `{callee}` carries a `borrow_until` obligation and must be \
                     bound to a variable; dropping it would detach the computation from the borrows it holds"
            ),
            Holder::InFlight(_) => format!(
                "the handle from `{callee}` was moved somewhere that cannot discharge its \
                     `borrow_until` obligation; only moving it into another local transfers the obligation"
            ),
        }
    }

    /// Call after `Expr`, `Return` and `Break` statements.
    pub fn finish_statement_obligations(&mut self) {
        let stray: Vec<usize> = self
            .borrow_obligations
            .iter()
            .enumerate()
            .filter(|(_, ob)| !ob.discharged && !matches!(ob.holder, Holder::Local(_)))
            .map(|(i, _)| i)
            .collect();

        for i in stray {
            let msg = self.outstanding_message(i, "here");
            self.record(TypeErrorKind::Generic(msg));
            self.discharge(i);
        }
    }

    /// Call after a `return` statement: dropping a handle never discharges.
    pub fn check_obligations_at_return(&mut self) {
        let left: Vec<usize> = self
            .borrow_obligations
            .iter()
            .enumerate()
            .filter(|(_, ob)| !ob.discharged && matches!(ob.holder, Holder::Local(_)))
            .map(|(i, _)| i)
            .collect();

        for i in left {
            let msg = self.outstanding_message(i, "at this `return`");
            self.record(TypeErrorKind::Generic(msg));
            self.discharge(i);
        }
    }

    /// Call at the end of `check_function`, before the borrow-checker scope is closed.
    pub fn finish_function_obligations(&mut self) {
        let left: Vec<usize> = self
            .borrow_obligations
            .iter()
            .enumerate()
            .filter(|(_, ob)| !ob.discharged && matches!(ob.holder, Holder::Local(_)))
            .map(|(i, _)| i)
            .collect();

        for i in left {
            let msg = self.outstanding_message(i, "at the end of the function");
            self.record(TypeErrorKind::Generic(msg));
            self.discharge(i);
        }
    }
}

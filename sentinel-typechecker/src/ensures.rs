use crate::TypeChecker;
use ir::{
    ast::{FuncSafety, MutabilityState},
    borrow_checker::Bound,
    errors::type_error::TypeErrorKind,
    hir::{
        HirExpr, HirFunc, HirParam, HirPattern, HirPatternKind, HirStmt, HirType, HirTypeKind,
        Operator, RefKind, StrId,
    },
    ir_hasher::FxHashMap,
};

/// A bound inside an `ensures` range.
/// `Param`/`ParamLen` carry a constant offset so that `path.len + 1` and `n - 2` are expressible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactBound {
    Const(i64),
    Param(usize, i64),
    ParamLen(usize, i64),
    Binder(StrId),
}

/// What an init fact is about: an argument, or the function's (named) return value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FactTarget {
    Param(usize),
    /// The return binder (`func f(): cbuf <- T`) whose name is the payload.
    Ret(StrId),
}

#[derive(Clone, Debug)]
pub struct VariantGuard {
    pub variant: StrId,
    pub bindings: Vec<(StrId, StrId)>,
    pub next: Option<(StrId, Box<VariantGuard>)>,
}

/// (variant, field) steps from the return value down to the binder's value.
/// Empty means the binder is the whole return value.
type BinderRoute = Vec<(StrId, StrId)>;

struct GuardLevel {
    variant: StrId,
    bindings: Vec<(StrId, StrId)>,
    via: Option<StrId>,
}

#[derive(Clone, Debug)]
pub struct EnsuresFact {
    pub guard: Option<VariantGuard>,
    pub target: FactTarget,
    /// `true` for `place != uninit`, `false` for `place == uninit`.
    pub init: bool,
    pub path: Vec<StrId>,
    pub lo: FactBound,
    pub hi: FactBound,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InstBound {
    Const(i64),
    Var(StrId),
    Binder(StrId),
}

/// Caller-side instance of a fact, in terms of caller locals.
#[derive(Clone, Debug)]
pub struct InstFact {
    pub guard: Option<VariantGuard>,
    pub root: StrId,
    pub path: Vec<StrId>,
    pub lo: InstBound,
    pub hi: InstBound,
    pub len: Option<usize>,
}

/// What a callee promised about (a path inside) its return value, resolved for one call site.
#[derive(Clone, Debug)]
pub struct RetInit {
    pub path: Vec<StrId>,
    /// `true` = `!= uninit`, `false` = `== uninit`.
    pub init: bool,
    /// `None` when the bounds weren't constants at this call site.
    pub range: Option<(i64, i64)>,
}

/// Arm-scoped fact: `root.path[lo..hi]` is initialized.
#[derive(Clone, Debug)]
pub struct SymInit {
    pub root: StrId,
    pub path: Vec<StrId>,
    pub lo: InstBound,
    pub hi: InstBound,
}

enum InitPred<'e, 'a, 'bump> {
    /// `place != uninit` (true) or `place == uninit` (false).
    Target(&'e HirExpr<'a, 'bump>, bool),
    /// `place <op> uninit` with an operator other than `==` / `!=`.
    WrongOp,
    /// Any other expression: an arbitrary boolean condition.
    Arbitrary,
}

type ParamMap<'a, 'bump> = FxHashMap<StrId, (usize, HirType<'a, 'bump>)>;

/// A relational `ensures` conjunct the checker can reason about without a solver:
/// `lhs == rhs` or `lhs != rhs`, where both sides are integer expressions built from literals,
/// parameters and `+ - *` with a literal (e.g. `i != j`, `a + 1 == b`).
///
/// These are *verified* at every exit of a safe function against the facts the borrow checker
/// already holds at that point (enclosing `if` conditions, facts from earlier calls' `ensures`),
/// and *assumed* by callers after the call, with the parameters replaced by the arguments.
/// Nothing is solved: a fact is only ever proven by being, or being a constant shift of, a
/// fact that is already known.
struct RelCond<'e, 'a, 'bump> {
    lhs: &'e HirExpr<'a, 'bump>,
    rhs: &'e HirExpr<'a, 'bump>,
    equal: bool,
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    fn as_binary<'e>(
        e: &'e HirExpr<'a, 'bump>,
    ) -> Option<(&'e HirExpr<'a, 'bump>, &'e Operator, &'e HirExpr<'a, 'bump>)> {
        match e {
            HirExpr::Binary {
                left, op, right, ..
            }
            | HirExpr::Comparison {
                left, op, right, ..
            } => Some((&**left, op, &**right)),
            _ => None,
        }
    }

    fn split_conjuncts<'e>(e: &'e HirExpr<'a, 'bump>, out: &mut Vec<&'e HirExpr<'a, 'bump>>) {
        match Self::as_binary(e) {
            Some((l, Operator::LogicalAnd, r)) => {
                Self::split_conjuncts(l, out);
                Self::split_conjuncts(r, out);
            }
            _ => out.push(e),
        }
    }

    fn init_predicate<'e>(e: &'e HirExpr<'a, 'bump>) -> InitPred<'e, 'a, 'bump> {
        match Self::as_binary(e) {
            Some((l, op, HirExpr::Uninit { .. })) => match op {
                Operator::NotEquals => InitPred::Target(l, true),
                Operator::Equals => InitPred::Target(l, false),
                _ => InitPred::WrongOp,
            },
            _ => InitPred::Arbitrary,
        }
    }

    /// Tiny constant evaluator: the only `ensures` conditions the checker can prove on its own
    /// besides init facts it verifies at `return`.
    fn const_bool(e: &HirExpr<'a, 'bump>) -> Option<bool> {
        match e {
            HirExpr::Boolean(b, _) => Some(*b),
            HirExpr::Comparison {
                left, op, right, ..
            } => {
                let (HirExpr::Number(a, _), HirExpr::Number(b, _)) = (&**left, &**right) else {
                    return None;
                };
                Some(match op {
                    Operator::Equals => a == b,
                    Operator::NotEquals => a != b,
                    Operator::LessThan => a < b,
                    Operator::LessThanOrEqual => a <= b,
                    Operator::GreaterThan => a > b,
                    Operator::GreaterThanOrEqual => a >= b,
                    _ => return None,
                })
            }
            _ => None,
        }
    }

    fn param_map(func: &HirFunc<'a, 'bump>) -> ParamMap<'a, 'bump> {
        let mut params: ParamMap<'a, 'bump> = FxHashMap::default();
        let mut idx = 0;
        for p in func.params.unwrap_or(&[]) {
            if let HirParam::Normal {
                name, param_type, ..
            } = p
            {
                params.insert(*name, (idx, *param_type));
                idx += 1;
            }
        }
        params
    }

    fn is_trusted(func: &HirFunc<'a, 'bump>) -> bool {
        func.body.is_none() || matches!(func.function_metadata.func_safety, FuncSafety::Unsafe)
    }

    /// Operands of a relational `ensures`: literals, parameters, and `+ - *` of those.
    fn rel_operand_ok(e: &HirExpr<'a, 'bump>, params: &ParamMap<'a, 'bump>) -> bool {
        match e {
            HirExpr::Number(..) => true,
            HirExpr::Ident(n, _) => params.contains_key(n),
            HirExpr::Binary {
                left, op, right, ..
            } if matches!(op, Operator::Add | Operator::Subtract | Operator::Multiply) => {
                Self::rel_operand_ok(&**left, params) && Self::rel_operand_ok(&**right, params)
            }
            _ => false,
        }
    }

    /// Decides whether one `ensures` conjunct is a relational condition (see [`RelCond`]).
    /// The `Err` text says why not; it ends up in the "can't prove" error.
    fn classify_rel<'e>(
        &mut self,
        c: &'e HirExpr<'a, 'bump>,
        has_pattern: bool,
        binders: &[StrId],
        params: &ParamMap<'a, 'bump>,
    ) -> Result<RelCond<'e, 'a, 'bump>, String> {
        if has_pattern {
            return Err("conditions under `ensures case` aren't checked".into());
        }
        for b in binders {
            if self.expr_references_local(c, *b) {
                return Err(
                    "it mentions a return binding, only conditions over parameters can be checked"
                        .into(),
                );
            }
        }
        let Some((l, op, r)) = Self::as_binary(c) else {
            return Err(
                "only `==` / `!=` comparisons between integer expressions are supported".into(),
            );
        };
        let equal = match op {
            Operator::Equals => true,
            Operator::NotEquals => false,
            _ => {
                return Err(
                    "only `==` / `!=` comparisons between integer expressions are supported".into(),
                );
            }
        };
        if !Self::rel_operand_ok(l, params) || !Self::rel_operand_ok(r, params) {
            return Err(
                "operands may only be integer literals, parameters, and `+` `-` `*` of those"
                    .into(),
            );
        }
        Ok(RelCond {
            lhs: l,
            rhs: r,
            equal,
        })
    }

    /// Every relational conjunct of `func`'s `ensures` clauses. Conjuncts that are init
    /// predicates, constant, or unsupported are skipped (the in-scope pass of `analyze_ensures`
    /// is what reports on those).
    fn relational_conditions<'e>(
        &mut self,
        func: &'e HirFunc<'a, 'bump>,
    ) -> Vec<RelCond<'e, 'a, 'bump>> {
        let mut out = Vec::new();
        if func.ensures.is_empty() {
            return out;
        }
        let params = Self::param_map(func);
        let binders: Vec<StrId> = func.return_binders.iter().map(|b| b.name).collect();
        for clause in func.ensures.iter() {
            let mut conj = Vec::new();
            Self::split_conjuncts(&clause.condition, &mut conj);
            for c in conj {
                if !matches!(Self::init_predicate(c), InitPred::Arbitrary)
                    || Self::const_bool(c).is_some()
                {
                    continue;
                }
                if let Ok(rc) = self.classify_rel(c, clause.pattern.is_some(), &binders, &params) {
                    out.push(rc);
                }
            }
        }
        out
    }

    fn identity_subst(params: &ParamMap<'a, 'bump>) -> FxHashMap<StrId, Bound> {
        params.keys().map(|n| (*n, Bound::Symbol(*n))).collect()
    }

    fn bound_offset(b: Bound, c: i64) -> Option<Bound> {
        match b {
            Bound::Const(x) => x.checked_add(c).map(Bound::Const),
            Bound::Offset { base, offset } => offset
                .checked_add(c)
                .map(|offset| Bound::Offset { base, offset }),
            other if c == 0 => Some(other),
            other => Some(Bound::Offset {
                base: Box::new(other),
                offset: c,
            }),
        }
    }

    /// Builds the symbolic value of an `ensures` operand, replacing each parameter by
    /// `subst[param]`. Only literals, substituted parameters and `+ - *` with a literal are
    /// understood; anything else is `None` (never an `Opaque`, so nothing is ever "provable"
    /// by accident).
    fn ensures_bound(e: &HirExpr<'a, 'bump>, subst: &FxHashMap<StrId, Bound>) -> Option<Bound> {
        match e {
            HirExpr::Number(v, _) => Some(Bound::Const(*v)),
            HirExpr::Ident(n, _) => subst.get(n).cloned(),
            HirExpr::Binary {
                left, op, right, ..
            } => {
                let l = Self::ensures_bound(&**left, subst)?;
                let r = Self::ensures_bound(&**right, subst)?;
                match op {
                    Operator::Add => match (l, r) {
                        (Bound::Const(a), Bound::Const(b)) => a.checked_add(b).map(Bound::Const),
                        (b, Bound::Const(c)) | (Bound::Const(c), b) => Self::bound_offset(b, c),
                        _ => None,
                    },
                    Operator::Subtract => match (l, r) {
                        (Bound::Const(a), Bound::Const(b)) => a.checked_sub(b).map(Bound::Const),
                        (b, Bound::Const(c)) => Self::bound_offset(b, c.checked_neg()?),
                        _ => None,
                    },
                    Operator::Multiply => match (l, r) {
                        (Bound::Const(a), Bound::Const(b)) => a.checked_mul(b).map(Bound::Const),
                        (b, Bound::Const(c)) | (Bound::Const(c), b) => Some(Bound::Scale {
                            base: Box::new(b),
                            factor: c,
                        }),
                        _ => None,
                    },
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Can facts about `b` be tracked? A `mut` variable is fine when `allow_mut` (reassignment
    /// kills the facts about it, see `kill_facts_about`) unless something can write to it behind
    /// the checker's back (`addr_taken`: a `&mut` to it exists, or a closure assigns it).
    /// Callee-side checks pass `allow_mut = false`: an `ensures` speaks about a parameter's value
    /// at exit, which for a `mut` parameter is not the value the caller passed in.
    pub(crate) fn bound_ok(&self, b: &Bound, allow_mut: bool) -> bool {
        match b {
            Bound::Const(_) => true,
            Bound::Symbol(n) => {
                self.context.get_variable(n).is_some()
                    && if allow_mut {
                        !self.addr_taken.contains(n)
                    } else {
                        !self.context.is_mutable(n)
                    }
            }
            Bound::Offset { base, .. } | Bound::Scale { base, .. } => {
                self.bound_ok(base, allow_mut)
            }
            Bound::Sum(a, c) => self.bound_ok(a, allow_mut) && self.bound_ok(c, allow_mut),
            Bound::Min(v) | Bound::Max(v) => v.iter().all(|x| self.bound_ok(x, allow_mut)),
            Bound::SelfField(_) | Bound::FreshPerCall | Bound::Opaque(_) => false,
        }
    }

    /// Both sides of a relational condition as bounds under `subst`, or why that isn't possible.
    fn rel_bounds(
        &self,
        rc: &RelCond<'_, 'a, 'bump>,
        subst: &FxHashMap<StrId, Bound>,
        allow_mut: bool,
    ) -> Result<(Bound, Bound), String> {
        let (Some(l), Some(r)) = (
            Self::ensures_bound(rc.lhs, subst),
            Self::ensures_bound(rc.rhs, subst),
        ) else {
            return Err(
                "operands must be integer expressions of parameters with literal offsets".into(),
            );
        };
        if !self.bound_ok(&l, allow_mut) || !self.bound_ok(&r, allow_mut) {
            return Err(
                "it involves a `mut` variable, whose value can change before this is used".into(),
            );
        }
        Ok((l, r))
    }

    /// Does `b` mention any of `names`?
    pub(crate) fn bound_mentions(b: &Bound, names: &[StrId]) -> bool {
        match b {
            Bound::Symbol(n) => names.contains(n),
            Bound::Offset { base, .. } | Bound::Scale { base, .. } => {
                Self::bound_mentions(base, names)
            }
            Bound::Sum(a, c) => Self::bound_mentions(a, names) || Self::bound_mentions(c, names),
            Bound::Min(v) | Bound::Max(v) => v.iter().any(|x| Self::bound_mentions(x, names)),
            Bound::Const(_) | Bound::SelfField(_) | Bound::FreshPerCall | Bound::Opaque(_) => false,
        }
    }

    fn show_bound(b: &Bound) -> String {
        match b {
            Bound::Const(c) => c.to_string(),
            Bound::Symbol(n) => n.to_string(),
            Bound::Offset { base, offset } if *offset < 0 => {
                format!("{} - {}", Self::show_bound(base), -offset)
            }
            Bound::Offset { base, offset } => format!("{} + {}", Self::show_bound(base), offset),
            Bound::Scale { base, factor } => format!("({}) * {}", Self::show_bound(base), factor),
            Bound::Sum(a, c) => format!("{} + {}", Self::show_bound(a), Self::show_bound(c)),
            _ => "_".into(),
        }
    }

    /// Verifies the relational `ensures` conditions of a safe function at one exit (a `return`,
    /// or falling off the end of a `void` function). Each one must already be known from the
    /// facts the borrow checker holds here. `unsafe` / bodiless functions are trusted.
    ///
    /// Conjuncts that can't be classified were already reported by `analyze_ensures`, so they
    /// are skipped silently here.
    pub(crate) fn check_ensures_conditions_at_exit(&mut self, func: &HirFunc<'a, 'bump>) {
        if Self::is_trusted(func) {
            return;
        }
        let params = Self::param_map(func);
        let subst = Self::identity_subst(&params);

        // The function's own conditions, then those of every interface method it implements:
        // callers of the interface only see the latter, so each implementation must prove them.
        let mut work: Vec<(HirFunc<'a, 'bump>, FxHashMap<StrId, Bound>)> = Vec::new();
        if !func.ensures.is_empty() {
            work.push((*func, subst));
        }
        for m in self.implemented_interface_methods(func) {
            if m.ensures.is_empty() {
                continue;
            }
            // Interface parameter names -> this implementation's parameters, by position.
            let iface_params = Self::param_map(&m);
            let mut map: FxHashMap<StrId, Bound> = FxHashMap::default();
            for (iname, (idx, _)) in iface_params.iter() {
                if let Some((own, _)) = params.iter().find(|(_, (i, _))| i == idx) {
                    map.insert(*iname, Bound::Symbol(*own));
                }
            }
            work.push((m, map));
        }

        for (f, subst) in work {
            let conds = self.relational_conditions(&f);
            for rc in conds {
                let Ok((l, r)) = self.rel_bounds(&rc, &subst, false) else {
                    continue;
                };
                let proven = if rc.equal {
                    self.borrow_checker.proves_equal(&l, &r)
                } else {
                    self.borrow_checker.proves_not_equal(&l, &r)
                };
                if !proven {
                    self.record(TypeErrorKind::Generic(format!(
                        "can't prove `ensures {} {} {}` on this path: it must follow from a \
                         condition checked on the way here, or from an earlier call's `ensures`{}",
                        Self::show_bound(&l),
                        if rc.equal { "==" } else { "!=" },
                        Self::show_bound(&r),
                        if std::ptr::eq(f.ensures.as_ptr(), func.ensures.as_ptr()) {
                            ""
                        } else {
                            " (required by the interface method this implements)"
                        },
                    )));
                }
            }
        }
    }

    /// The interface methods `func` implements: same name, on an interface its `impl_target`
    /// type implements.
    fn implemented_interface_methods(&self, func: &HirFunc<'a, 'bump>) -> Vec<HirFunc<'a, 'bump>> {
        let Some(target) = func.impl_target else {
            return Vec::new();
        };
        let Some(ifaces) = self.context.get_struct_interfaces(target) else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for name in ifaces {
            let Some(iface) = self.context.get_interface(*name) else {
                continue;
            };
            for m in iface.methods.unwrap_or(&[]) {
                if m.unmangled_name == func.unmangled_name {
                    out.push(*m);
                }
            }
        }
        out
    }

    /// Call where a `void` function's body can fall through to its end (the body's last
    /// statement doesn't diverge). Functions with a return type exit through `return`.
    pub(crate) fn check_ensures_at_fallthrough(&mut self, func: &HirFunc<'a, 'bump>) {
        let is_void = func
            .return_type
            .map_or(true, |t| matches!(t.kind, HirTypeKind::Void));
        if is_void {
            self.check_ensures_conditions_at_exit(func);
        }
    }

    /// Called from `register_call_ensures`: resolve this call's relational conditions against
    /// its arguments and keep them (keyed by the call) until the statement that contains the
    /// call applies them with [`Self::apply_call_rel_facts`].
    fn stash_rel_ensures(
        &mut self,
        call: &HirExpr<'a, 'bump>,
        func: &HirFunc<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
    ) {
        if func.ensures.is_empty() {
            return;
        }
        let params = Self::param_map(func);
        let base = Self::arg_base(func, args);
        let mut subst: FxHashMap<StrId, Bound> = FxHashMap::default();
        for (name, (idx, _)) in params.iter() {
            let Some(arg) = args.get(base + *idx) else {
                continue;
            };
            if matches!(arg, HirExpr::Ref { .. }) {
                continue;
            }
            let b = self.expr_to_bound(arg);
            subst.insert(*name, b);
        }

        let conds = self.relational_conditions(func);
        let mut stash: Vec<(Bound, Bound, bool)> = Vec::new();
        for rc in conds {
            let Ok((l, r)) = self.rel_bounds(&rc, &subst, true) else {
                continue;
            };
            if matches!((&l, &r), (Bound::Const(_), Bound::Const(_))) {
                continue; // nothing to learn: already decided by the constants
            }
            stash.push((l, r, rc.equal));
        }
        if !stash.is_empty() {
            self.pending_rel.insert(Self::expr_key(call), stash);
        }
    }

    /// Records, for the rest of the enclosing scope, what the callee promised about the
    /// arguments of `call`. Callers go through `apply_unconditional_call_facts`, which only
    /// reaches calls that are certain to run: a call in a closure or the right side of `&&` /
    /// `||` may never run, so its promises must not be assumed.
    pub(crate) fn apply_call_rel_facts(&mut self, call: &HirExpr<'a, 'bump>) {
        let Some(facts) = self.pending_rel.remove(&Self::expr_key(call)) else {
            return;
        };
        for (l, r, equal) in facts {
            if equal {
                self.borrow_checker.assume_equal_scoped(l, r);
            } else {
                self.borrow_checker.assume_not_equal_scoped(l, r);
            }
        }
    }

    /// `let name = fn (..) ensures .. { .. };` (immutable): remember the lambda's function under
    /// the binding, so calls to `name` get its `ensures`. Call right after `name` is declared.
    pub(crate) fn bind_lambda_ensures(
        &mut self,
        name: StrId,
        value: &HirExpr<'a, 'bump>,
        mutable: bool,
    ) {
        self.lambda_bindings.remove(&name);
        if mutable {
            return;
        }
        let HirExpr::Lambda { body, .. } = value else {
            return;
        };
        let Some(f) = self.lambda_funcs.get(&Self::stmt_key(body)).copied() else {
            return;
        };
        let Some((sym, _)) = self.context.get_variable(name) else {
            return;
        };
        self.lambda_bindings.insert(name, (sym, f));
    }

    /// `f(args)` where `f` is an immutable local bound to a lambda with `ensures`: resolve the
    /// lambda's conditions against `args`, exactly like a call of a named function. The symbol
    /// check makes sure `f` is still that binding and not a later variable (or a function) that
    /// reuses the name.
    pub(crate) fn stash_lambda_call(&mut self, call: &HirExpr<'a, 'bump>) {
        let HirExpr::Call { callee, args, .. } = call else {
            return;
        };
        let HirExpr::Ident(n, _) = &**callee else {
            return;
        };
        let Some((sym, f)) = self.lambda_bindings.get(n).copied() else {
            return;
        };
        if self.context.get_variable(*n).map(|(s, _)| s) != Some(sym) {
            return;
        }
        self.register_call_ensures(call, &f, args);
    }

    /// `name` was reassigned, rebound or can now be written through a reference: forget every
    /// recorded `==` / `!=` fact that mentions it.
    pub(crate) fn kill_facts_about(&mut self, name: StrId) {
        self.borrow_checker.kill_facts_where(&|n| n == name);
        let names = [name];
        for facts in self.pending_rel.values_mut() {
            facts.retain(|(l, r, _)| {
                !Self::bound_mentions(l, &names) && !Self::bound_mentions(r, &names)
            });
        }
    }

    /// Forget every fact about a `mut` (or address-taken) variable. Run on entry to a loop: the
    /// body is walked once, but it runs again after assignments later in the body.
    pub(crate) fn kill_facts_about_mutables(&mut self) {
        let ctx = &self.context;
        let taken = &self.addr_taken;
        self.borrow_checker
            .kill_facts_where(&|n| ctx.is_mutable(n) || taken.contains(&n));
    }

    /// `&mut x` (or `&alias x`) of a local, or an assignment to a captured local inside a
    /// closure: from now on `x` can change without a visible assignment, so no fact about it is
    /// trusted any more.
    pub(crate) fn note_written_behind_back(&mut self, name: StrId) {
        self.addr_taken.insert(name);
        self.kill_facts_about(name);
    }

    /// Is a branch-condition fact usable? Not if it involves a variable that can be written
    /// behind the checker's back.
    pub(crate) fn fact_is_trackable(&self, l: &Bound, r: &Bound) -> bool {
        let taken: Vec<StrId> = self.addr_taken.iter().copied().collect();
        taken.is_empty() || !(Self::bound_mentions(l, &taken) || Self::bound_mentions(r, &taken))
    }

    /// Applies the promises of every call in `e` that is certain to run whenever `e` is
    /// evaluated: the call itself, calls among its arguments, operands of arithmetic and
    /// comparisons, and the left side of `&&` / `||`. The right side of `&&` / `||` and closure
    /// bodies may never run, so they are skipped.
    pub(crate) fn apply_unconditional_call_facts(&mut self, e: &HirExpr<'a, 'bump>) {
        match e {
            HirExpr::Binary {
                left, op, right, ..
            }
            | HirExpr::Comparison {
                left, op, right, ..
            } => {
                self.apply_unconditional_call_facts(left);
                if !matches!(op, Operator::LogicalAnd | Operator::LogicalOr) {
                    self.apply_unconditional_call_facts(right);
                }
            }
            HirExpr::Call { args, .. } | HirExpr::InterfaceCall { args, .. } => {
                for a in args.iter() {
                    self.apply_unconditional_call_facts(a);
                }
                self.apply_call_rel_facts(e);
            }
            _ => {}
        }
    }

    /// Like `apply_call_rel_facts`, but leaves the stash in place so a later path can apply the
    /// same call's facts again (the call ran on every path that reaches it).
    fn apply_call_rel_facts_keep(&mut self, call: &HirExpr<'a, 'bump>) {
        let Some(facts) = self.pending_rel.get(&Self::expr_key(call)).cloned() else {
            return;
        };
        for (l, r, equal) in facts {
            if equal {
                self.borrow_checker.assume_equal_scoped(l, r);
            } else {
                self.borrow_checker.assume_not_equal_scoped(l, r);
            }
        }
    }

    /// On the path where `cond` evaluated to `holds`, which of its calls certainly ran?
    /// `a && b` being true means both ran; `a || b` being false means both ran. (The calls that
    /// always run were already applied by `apply_unconditional_call_facts`.) Called from the
    /// branch scopes of an `if`, so the facts end with the branch.
    pub(crate) fn apply_path_call_facts(&mut self, cond: &HirExpr<'a, 'bump>, holds: bool) {
        match cond {
            HirExpr::Binary {
                left, op, right, ..
            } if matches!(
                (op, holds),
                (Operator::LogicalAnd, true) | (Operator::LogicalOr, false)
            ) =>
            {
                self.apply_path_call_facts(left, holds);
                self.apply_path_call_facts(right, holds);
            }
            HirExpr::Binary {
                left, op, right, ..
            }
            | HirExpr::Comparison {
                left, op, right, ..
            } => {
                self.apply_path_call_facts(left, holds);
                if !matches!(op, Operator::LogicalAnd | Operator::LogicalOr) {
                    self.apply_path_call_facts(right, holds);
                }
            }
            HirExpr::Call { args, .. } | HirExpr::InterfaceCall { args, .. } => {
                for a in args.iter() {
                    self.apply_path_call_facts(a, holds);
                }
                self.apply_call_rel_facts_keep(cond);
            }
            _ => {}
        }
    }

    /// Length of a fixed-size array reached by walking `root.path` through the declared types of
    /// a parameter or whole-return binding. Purely type-driven, so it also works in the silent
    /// syntactic pass (`in_scope = false`) where no locals are in scope.
    fn static_array_len(
        &self,
        object: &HirExpr<'a, 'bump>,
        params: &ParamMap<'a, 'bump>,
        rets: &[(StrId, HirType<'a, 'bump>)],
    ) -> Option<usize> {
        let (root, path) = self.static_field_path(object)?;
        let mut ty = match params.get(&root) {
            Some(&(_, t)) => t,
            None => rets.iter().find(|(n, _)| *n == root)?.1,
        };
        for field in &path {
            let t = Self::peel_indirections(*Self::strip_ref(&ty));
            let HirTypeKind::Struct { name, .. } = t.kind else {
                return None;
            };
            let def = self.context.get_struct(name)?;
            ty = def.fields.iter().find(|f| f.name == *field)?.field_type;
        }
        match Self::strip_ref(&ty).kind {
            HirTypeKind::Array(_, l) => Some(l),
            _ => None,
        }
    }

    /// `x.len` / `x.len()` where `x` is a parameter (symbolic) or a fixed-size array reachable
    /// from a parameter / return binding (constant).
    fn len_bound(
        &self,
        object: &HirExpr<'a, 'bump>,
        params: &ParamMap<'a, 'bump>,
        rets: &[(StrId, HirType<'a, 'bump>)],
    ) -> Result<FactBound, String> {
        if let HirExpr::Ident(p, _) = object {
            if let Some((i, _)) = params.get(p) {
                return Ok(FactBound::ParamLen(*i, 0));
            }
        }
        if let Some(l) = self.static_array_len(object, params, rets) {
            return Ok(FactBound::Const(l as i64));
        }
        Err("`.len` in `ensures` must be taken on a parameter, or on a fixed-size array field of a parameter or return binding".into())
    }

    fn fact_bound(
        &self,
        e: &HirExpr<'a, 'bump>,
        params: &ParamMap<'a, 'bump>,
        rets: &[(StrId, HirType<'a, 'bump>)],
        bound: &[StrId],
    ) -> Result<FactBound, String> {
        match e {
            HirExpr::Number(i, _) => Ok(FactBound::Const(*i)),
            HirExpr::Ident(n, _) if bound.contains(n) => Ok(FactBound::Binder(*n)),
            HirExpr::Ident(n, _) => params
                .get(n)
                .map(|(i, _)| FactBound::Param(*i, 0))
                .ok_or_else(|| format!("`{}` is neither a parameter nor an `ensures` binding", n)),
            HirExpr::Binary {
                left, op, right, ..
            } if matches!(op, Operator::Add | Operator::Subtract) => {
                let HirExpr::Number(k, _) = &**right else {
                    return Err("bounds in `ensures` may only add/subtract an integer literal".into());
                };
                let k = if matches!(op, Operator::Subtract) { -*k } else { *k };
                match self.fact_bound(&**left, params, rets, bound)? {
                    FactBound::Const(c) => Ok(FactBound::Const(c + k)),
                    FactBound::Param(i, o) => Ok(FactBound::Param(i, o + k)),
                    FactBound::ParamLen(i, o) => Ok(FactBound::ParamLen(i, o + k)),
                    FactBound::Binder(_) => {
                        Err("arithmetic on `ensures` bindings isn't supported".into())
                    }
                }
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. }
                if field.as_str() == "len" =>
            {
                self.len_bound(&**object, params, rets)
            }
            HirExpr::Call { callee, args, .. } if args.is_empty() => match &**callee {
                HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. }
                    if field.as_str() == "len" =>
                {
                    self.len_bound(&**object, params, rets)
                }
                _ => Err("only `.len()` calls are supported in `ensures` bounds".into()),
            },
            _ => Err("unsupported bound in `ensures`: use integer literals, parameters, `param.len`, `+`/`-` a literal, or bindings".into()),
        }
    }

    fn resolve_return_binder(
        &self,
        ret: Option<HirType<'a, 'bump>>,
        path: &[usize],
    ) -> Result<(BinderRoute, HirType<'a, 'bump>), String> {
        let Some(mut cur) = ret else {
            return Err("return binder on a function without a return type".into());
        };
        let mut route = Vec::with_capacity(path.len());
        for &idx in path {
            let HirTypeKind::Enum {
                name, type_args, ..
            } = cur.kind
            else {
                return Err(
                    "return binders inside generic arguments are only supported on enums".into(),
                );
            };
            let def = self
                .context
                .get_enum(name)
                .ok_or("unknown enum in return type")?;
            let gname = def
                .generics
                .and_then(|g| g.get(idx))
                .map(|g| g.name)
                .ok_or("return binder names a generic argument that doesn't exist")?;
            let (mut hit, mut count) = (None, 0);
            for v in def.variants.iter() {
                for f in v.fields.iter() {
                    if matches!(f.field_type.kind, HirTypeKind::Generic(n) if n == gname) {
                        count += 1;
                        hit = Some((v.name, f.name));
                    }
                }
            }
            match (count, hit) {
                (1, Some(h)) => route.push(h),
                (0, _) => return Err("that generic argument isn't stored in any variant".into()),
                _ => {
                    return Err(
                        "that generic argument appears in several variants; use `ensures case ...`"
                            .into(),
                    );
                }
            }
            cur = *type_args.get(idx).ok_or("missing generic argument")?;
        }
        Ok((route, cur))
    }

    fn merge_routes(routes: &[(StrId, &[(StrId, StrId)])]) -> Result<Option<VariantGuard>, String> {
        const CONFLICT: &str = "an `ensures` clause can't use bindings from different variants (only one is ever live)";
        let mut levels: Vec<GuardLevel> = Vec::new();
        let mut rooted = false;
        for &(name, route) in routes {
            if route.is_empty() {
                rooted = true;
                continue;
            }
            for (k, &(variant, field)) in route.iter().enumerate() {
                if k == levels.len() {
                    levels.push(GuardLevel {
                        variant,
                        bindings: Vec::new(),
                        via: None,
                    });
                }
                let lvl = &mut levels[k];
                if lvl.variant != variant {
                    return Err(CONFLICT.into());
                }
                if k + 1 == route.len() {
                    lvl.bindings.push((field, name));
                } else {
                    match lvl.via {
                        Some(f) if f != field => return Err(CONFLICT.into()),
                        _ => lvl.via = Some(field),
                    }
                }
            }
        }
        if rooted && !levels.is_empty() {
            return Err(CONFLICT.into());
        }
        let mut next: Option<VariantGuard> = None;
        for lvl in levels.into_iter().rev() {
            let via = lvl.via;
            next = Some(VariantGuard {
                variant: lvl.variant,
                bindings: lvl.bindings,
                next: next.and_then(|g| via.map(|f| (f, Box::new(g)))),
            });
        }
        Ok(next)
    }

    pub fn enum_variant_pattern(
        pat: &HirPattern<'a, 'bump>,
        enum_ty: &HirType<'a, 'bump>,
    ) -> Option<(StrId, Vec<(StrId, StrId)>)> {
        let HirPatternKind::EnumVariant {
            variant, bindings, ..
        } = pat.kind
        else {
            return None;
        };
        let HirTypeKind::Enum { variants, .. } = Self::strip_ref(enum_ty).kind else {
            return None;
        };
        let def = variants.iter().find(|v| v.name == variant)?;
        let pairs = def
            .fields
            .iter()
            .zip(bindings.iter())
            .filter(|(_, b)| b.as_str() != "_")
            .map(|(f, b)| (f.name, *b))
            .collect();
        Some((variant, pairs))
    }

    pub(crate) fn ensures_facts_for(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<EnsuresFact> {
        if let Some(c) = self.ensures_facts_cache.get(&func.name) {
            return c.clone();
        }
        let facts = self.analyze_ensures(func, false);
        self.ensures_facts_cache.insert(func.name, facts.clone());
        facts
    }

    /// Call with `in_scope = true` from the function-check entry, after the params are in scope:
    /// that run type-checks and reports. `false` is silent and purely syntactic (callers use it).
    ///
    /// Clause conditions are split on `&&` and each conjunct is one of:
    ///   * `place != uninit` / `place == uninit`  -> an init fact (param or return binder),
    ///   * anything else                          -> an arbitrary boolean expression.
    /// Arbitrary expressions must type-check as `bool`; the checker can only prove constant
    /// ones, so in a safe function anything else is rejected. `unsafe` functions are trusted.
    /// Init facts on return binders are verified at `return` (`check_return_against_ensures`).
    pub fn analyze_ensures(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        in_scope: bool,
    ) -> Vec<EnsuresFact> {
        let mut out = Vec::new();
        if func.ensures.is_empty() {
            return out;
        }
        let trusted = Self::is_trusted(func);
        let params = Self::param_map(func);

        let binders: Vec<_> = func
            .return_binders
            .iter()
            .map(|b| {
                (
                    b.name,
                    b.span,
                    self.resolve_return_binder(func.return_type, b.arg_path),
                )
            })
            .collect();

        // Binders that name the *whole* return value: usable as the root of an init place.
        let rets: Vec<(StrId, HirType<'a, 'bump>)> = binders
            .iter()
            .filter_map(|(n, _, res)| match res {
                Ok((route, ty)) if route.is_empty() => Some((*n, *ty)),
                _ => None,
            })
            .collect();

        let binder_names: Vec<StrId> = binders.iter().map(|(n, _, _)| *n).collect();

        'clauses: for clause in func.ensures.iter() {
            if in_scope {
                self.set_span(clause.span);
            }

            let mut bound_names: Vec<StrId> = Vec::new();
            let guard: Option<VariantGuard> = if let Some(pat) = &clause.pattern {
                let Some(ret) = func.return_type else {
                    if in_scope {
                        self.record(TypeErrorKind::Generic(
                            "`ensures case` on a function without a return type".into(),
                        ));
                    }
                    continue 'clauses;
                };
                match Self::enum_variant_pattern(pat, &ret) {
                    Some((variant, bindings)) => {
                        bound_names.extend(bindings.iter().map(|(_, n)| *n));
                        Some(VariantGuard {
                            variant,
                            bindings,
                            next: None,
                        })
                    }
                    None => {
                        if in_scope {
                            self.record(TypeErrorKind::Generic(
                                "`ensures case` expects an enum variant pattern".into(),
                            ));
                        }
                        continue 'clauses;
                    }
                }
            } else {
                let mut routes: Vec<(StrId, &[(StrId, StrId)])> = Vec::new();
                for (name, _, res) in &binders {
                    if !self.expr_references_local(&clause.condition, *name) {
                        continue;
                    }
                    match res {
                        Ok((route, _)) => {
                            routes.push((*name, route.as_slice()));
                            bound_names.push(*name);
                        }
                        Err(msg) => {
                            if in_scope {
                                self.record(TypeErrorKind::Generic(msg.clone()));
                            }
                            continue 'clauses;
                        }
                    }
                }
                match Self::merge_routes(&routes) {
                    Ok(g) => g,
                    Err(msg) => {
                        if in_scope {
                            self.record(TypeErrorKind::Generic(msg));
                        }
                        continue 'clauses;
                    }
                }
            };

            let saved_ctx = if in_scope {
                let child = self.context.create_child_scope();
                let old = std::mem::replace(&mut self.context, child);
                for (name, _, res) in &binders {
                    if let Ok((_, ty)) = res {
                        let sym = self.mint_symbol_id();
                        self.context.add_variable(*name, *ty, sym);
                    }
                }
                if let (Some(pat), Some(ret)) = (&clause.pattern, func.return_type) {
                    self.check_pattern_against_type(pat, &ret);
                    self.register_pattern_bindings(
                        pat,
                        &ret,
                        crate::initialization::BindingMode::ByValue,
                        None,
                        None,
                    );
                }
                Some(old)
            } else {
                None
            };

            let mut conj = Vec::new();
            Self::split_conjuncts(&clause.condition, &mut conj);
            for c in conj {
                match Self::init_predicate(c) {
                    InitPred::WrongOp => {
                        if in_scope {
                            self.record(TypeErrorKind::Generic(
                                "`uninit` can only be compared with `==` or `!=` in `ensures`"
                                    .into(),
                            ));
                        }
                    }
                    InitPred::Arbitrary => {
                        if in_scope {
                            let ty = self.check_expr_suppressed(c);
                            if !matches!(ty.kind, HirTypeKind::Boolean | HirTypeKind::Unknown) {
                                self.record(TypeErrorKind::TypeMismatch {
                                    expected: "bool".into(),
                                    found: crate::naming::type_to_string(
                                        &ty,
                                        &self.context.string_pool,
                                    )
                                    .to_string(),
                                });
                            } else if !trusted && Self::const_bool(c) != Some(true) {
                                // A relational condition is accepted here and verified at every
                                // exit (`check_ensures_conditions_at_exit`); anything else can't
                                // be proven by the checker.
                                let rel = self
                                    .classify_rel(
                                        c,
                                        clause.pattern.is_some(),
                                        &binder_names,
                                        &params,
                                    )
                                    .and_then(|rc| {
                                        self.rel_bounds(&rc, &Self::identity_subst(&params), false)
                                    });
                                if let Err(why) = rel {
                                    self.record(TypeErrorKind::Generic(format!(
                                        "can't prove this `ensures` condition ({}): mark the function `unsafe` to have the compiler trust it",
                                        why
                                    )));
                                }
                            }
                        }
                    }
                    InitPred::Target(place, init) => {
                        match self.init_fact_from_place(
                            place,
                            &params,
                            &rets,
                            &bound_names,
                            &guard,
                            init,
                            in_scope,
                            trusted,
                        ) {
                            Ok(f) => out.push(f),
                            Err(msg) => {
                                if in_scope {
                                    self.record(TypeErrorKind::Generic(msg));
                                }
                            }
                        }
                    }
                }
            }

            if let Some(old) = saved_ctx {
                self.context = old;
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn init_fact_from_place(
        &mut self,
        place: &HirExpr<'a, 'bump>,
        params: &ParamMap<'a, 'bump>,
        rets: &[(StrId, HirType<'a, 'bump>)],
        bound: &[StrId],
        guard: &Option<VariantGuard>,
        init: bool,
        in_scope: bool,
        trusted: bool,
    ) -> Result<EnsuresFact, String> {
        let (object, lo_e, hi_e): (
            &HirExpr<'a, 'bump>,
            Option<&HirExpr<'a, 'bump>>,
            Option<&HirExpr<'a, 'bump>>,
        ) = match place {
            HirExpr::Slice {
                object, start, end, ..
            } => (&**object, Some(&**start), Some(&**end)),
            HirExpr::Index { object, index, .. } => (&**object, Some(&**index), None),
            other => (other, None, None),
        };

        let (root, path) = self.static_field_path(object).ok_or(
            "`ensures` initialization predicates must name a parameter or the return binding (or a field of one)",
        )?;

        let (target, root_ty) = if let Some(&(pidx, pty)) = params.get(&root) {
            (FactTarget::Param(pidx), pty)
        } else if let Some(&(_, rty)) = rets.iter().find(|(n, _)| *n == root) {
            (FactTarget::Ret(root), rty)
        } else {
            return Err(format!(
                "`{}` is neither a parameter nor a return binding of this function",
                root
            ));
        };

        if let FactTarget::Param(_) = target {
            let writable = matches!(
                root_ty.kind,
                HirTypeKind::Ref {
                    ref_kind: RefKind::Unique | RefKind::Alias,
                    ..
                } | HirTypeKind::SafePointer {
                    mutability_state: MutabilityState::Mut,
                    ..
                } | HirTypeKind::UnsafePointer {
                    mutability_state: MutabilityState::Mut,
                    ..
                }
            );
            if !writable {
                return Err(format!(
                    "`{}` can't be initialized by this function: it isn't a `&mut`/`&alias` or mutable pointer parameter",
                    root
                ));
            }
        }

        let (lo, hi) = match (lo_e, hi_e) {
            (Some(l), Some(h)) => (
                self.fact_bound(l, params, rets, bound)?,
                self.fact_bound(h, params, rets, bound)?,
            ),
            (Some(HirExpr::Number(i, _)), None) => (FactBound::Const(*i), FactBound::Const(*i + 1)),
            (Some(_), None) => {
                return Err("element predicates in `ensures` need a constant index".into());
            }
            (None, None) if path.is_empty() => match (target, root_ty.kind) {
                (FactTarget::Param(p), _) => (FactBound::Const(0), FactBound::ParamLen(p, 0)),
                (FactTarget::Ret(_), HirTypeKind::Array(_, l)) => {
                    (FactBound::Const(0), FactBound::Const(l as i64))
                }
                (FactTarget::Ret(_), _) => {
                    return Err(
                        "a whole-place predicate on the return binding needs an array type; write an explicit range like `x[0..<n]`".into(),
                    );
                }
            },
            _ => {
                return Err(
                    "whole-place init predicates are only supported on a parameter or return binding itself; write an explicit range like `x.field[0..<n]`".into(),
                );
            }
        };

        if in_scope {
            let obj_ty = self.check_expr_suppressed(object);
            if !matches!(
                Self::strip_ref(&obj_ty).kind,
                HirTypeKind::Array(..) | HirTypeKind::Slice(_)
            ) {
                return Err("`uninit` predicates in `ensures` need an array or slice place".into());
            }
            for e in [lo_e, hi_e].into_iter().flatten() {
                let t = self.check_expr_suppressed(e);
                if !self.is_integer(&t) {
                    return Err("`ensures` range bounds must be integers".into());
                }
            }
            // Facts about the return value are verified at `return`; facts about arguments
            // can't be verified yet, so they need an `unsafe` function.
            if matches!(target, FactTarget::Param(_)) && !trusted {
                return Err("can't verify this `ensures` yet: mark the function `unsafe` so the compiler trusts it, or initialize explicitly with `$assume_init`".into());
            }
        }

        Ok(EnsuresFact {
            guard: guard.clone(),
            target,
            init,
            path,
            lo,
            hi,
        })
    }

    /// Resolve a bound to a compile-time constant using only the function's own signature.
    fn const_fact_bound(b: FactBound, params: &ParamMap<'a, 'bump>) -> Option<i64> {
        match b {
            FactBound::Const(c) => Some(c),
            FactBound::ParamLen(j, off) => {
                let (_, t) = params.values().find(|(i, _)| *i == j)?;
                match Self::strip_ref(t).kind {
                    HirTypeKind::Array(_, l) => Some(l as i64 + off),
                    _ => None,
                }
            }
            _ => None,
        }
    }

    /// Call from the `return` check, before the returned expression is read.
    ///
    /// Returns `true` when `value` is a bare return binder that has `ensures` facts; the caller
    /// must then skip the normal "use of (possibly) uninitialized value" read check for it,
    /// because the facts *are* the statement of what is initialized. Unless the function is
    /// `unsafe`, each fact is verified here against the init state at the return point.
    pub(crate) fn check_return_against_ensures(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        value: Option<&HirExpr<'a, 'bump>>,
    ) -> bool {
        let facts: Vec<EnsuresFact> = self
            .ensures_facts_for(func)
            .into_iter()
            .filter(|f| matches!(f.target, FactTarget::Ret(_)))
            .collect();
        if facts.is_empty() {
            return false;
        }
        let Some(HirExpr::Ident(n, _)) = value else {
            return false;
        };
        let n = *n;
        if !facts.iter().any(|f| f.target == FactTarget::Ret(n)) {
            return false;
        }
        if Self::is_trusted(func) {
            return true;
        }
        let params = Self::param_map(func);
        let Some(node) = self.init_node(n) else {
            return true;
        };

        for f in facts.iter().filter(|f| f.target == FactTarget::Ret(n)) {
            if f.guard.is_some() {
                continue;
            }
            let shown: String = f.path.iter().map(|s| format!(".{}", s)).collect();
            let (Some(lo), Some(hi)) = (
                Self::const_fact_bound(f.lo, &params),
                Self::const_fact_bound(f.hi, &params),
            ) else {
                self.record(TypeErrorKind::Generic(format!(
                    "can't prove the `ensures` on `{}{}`: its bounds aren't compile-time constants; mark the function `unsafe` to trust it",
                    n, shown
                )));
                continue;
            };

            if lo >= hi {
                continue; // empty range: vacuously true
            }

            let target = Self::node_at_path_ref(&node, &f.path);
            let covered = match target {
                crate::initialization::InitNode::Array { ranges, .. } => {
                    ranges.contains_range(lo, hi)
                }
                crate::initialization::InitNode::Whole(
                    crate::initialization::InitStatus::Initialized,
                ) => true,
                _ => false,
            };
            let touched = match target {
                crate::initialization::InitNode::Array { ranges, .. } => {
                    ranges.ranges.iter().any(|&(s, e)| s < hi && lo < e)
                }
                crate::initialization::InitNode::Whole(
                    crate::initialization::InitStatus::Uninitialized,
                ) => false,
                _ => true,
            };

            if f.init && !covered {
                self.record(TypeErrorKind::Generic(format!(
                    "can't prove `ensures {}{}[{}..<{}] != uninit`: not every element is initialized on all paths to this `return`",
                    n, shown, lo, hi
                )));
            } else if !f.init && touched {
                self.record(TypeErrorKind::Generic(format!(
                    "can't prove `ensures {}{}[{}..<{}] == uninit`: part of that range may have been initialized",
                    n, shown, lo, hi
                )));
            }
        }
        true
    }

    fn arg_base(func: &HirFunc<'a, 'bump>, args: &[HirExpr<'a, 'bump>]) -> usize {
        let params = func.params.unwrap_or(&[]);
        let normal = params
            .iter()
            .filter(|p| matches!(p, HirParam::Normal { .. }))
            .count();
        let has_this = params.iter().any(|p| matches!(p, HirParam::This { .. }));
        if has_this && args.len() > normal {
            1
        } else {
            0
        }
    }

    fn const_inst_bound(
        &self,
        b: FactBound,
        args: &[HirExpr<'a, 'bump>],
        base: usize,
    ) -> Option<i64> {
        match b {
            FactBound::Const(c) => Some(c),
            FactBound::Binder(_) => None,
            FactBound::Param(j, off) => match args.get(base + j)? {
                HirExpr::Number(k, _) => Some(*k + off),
                _ => None,
            },
            FactBound::ParamLen(j, off) => match args.get(base + j)? {
                HirExpr::Ref { expr: e2, .. } => match &**e2 {
                    HirExpr::Slice { start, end, .. } => match (&**start, &**end) {
                        (HirExpr::Number(a, _), HirExpr::Number(b, _)) => Some(b - a + off),
                        _ => None,
                    },
                    other => match self.peek_type(other).kind {
                        HirTypeKind::Array(_, l) => Some(l as i64 + off),
                        _ => None,
                    },
                },
                _ => None,
            },
        }
    }

    /// Called from `register_call_ensures`: remember what the callee promised about its return
    /// value, resolved against this call's arguments, keyed by the call expression.
    fn stash_ret_ensures(
        &mut self,
        call: &HirExpr<'a, 'bump>,
        func: &HirFunc<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
    ) {
        let facts: Vec<EnsuresFact> = self
            .ensures_facts_for(func)
            .into_iter()
            .filter(|f| matches!(f.target, FactTarget::Ret(_)) && f.guard.is_none())
            .collect();
        if facts.is_empty() {
            return;
        }
        let base = Self::arg_base(func, args);
        let stash: Vec<RetInit> = facts
            .iter()
            .map(|f| RetInit {
                path: f.path.clone(),
                init: f.init,
                range: match (
                    self.const_inst_bound(f.lo, args, base),
                    self.const_inst_bound(f.hi, args, base),
                ) {
                    (Some(lo), Some(hi)) => Some((lo, hi)),
                    _ => None, // symbolic: stays uninit (conservative)
                },
            })
            .collect();
        self.pending_ret.insert(Self::expr_key(call), stash);
    }

    /// Call from the `let x = call(..)` handling, right after `x` is registered and initialized.
    /// The value is not wholly initialized: only the ranges the callee promised with
    /// `ensures ret.path[lo..<hi] != uninit` are. Everything else under those paths is marked
    /// uninitialized, so later reads must be justified.
    pub(crate) fn apply_ret_ensures(&mut self, local: StrId, call: &HirExpr<'a, 'bump>) {
        let Some(stash) = self.pending_ret.get(&Self::expr_key(call)).cloned() else {
            return;
        };
        let mut paths: Vec<Vec<StrId>> = Vec::new();
        for r in &stash {
            if !paths.contains(&r.path) {
                paths.push(r.path.clone());
            }
        }
        for p in &paths {
            self.mark_field_uninit(local, p);
        }
        for r in stash.iter().filter(|r| r.init) {
            if let Some((lo, hi)) = r.range {
                self.mark_array_range(local, &r.path, lo, hi, None);
            }
        }
    }

    pub(crate) fn register_call_ensures(
        &mut self,
        call: &HirExpr<'a, 'bump>,
        func: &HirFunc<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
    ) {
        self.stash_rel_ensures(call, func, args);
        let facts = self.ensures_facts_for(func);
        if facts.is_empty() {
            return;
        }
        self.stash_ret_ensures(call, func, args);
        let base = Self::arg_base(func, args);

        let mut pending = Vec::new();
        for f in &facts {
            if !f.init {
                continue; // `== uninit` facts add nothing for the caller to assume
            }
            let Some(inst) = self.instantiate_fact(f, args, base) else {
                continue;
            };
            match (&inst.guard, inst.lo, inst.hi) {
                (None, InstBound::Const(lo), InstBound::Const(hi)) => {
                    self.mark_array_range(inst.root, &inst.path, lo, hi, inst.len);
                }
                (None, _, _) => {} // unguarded symbolic facts aren't applied yet
                _ => pending.push(inst),
            }
        }
        if !pending.is_empty() {
            self.pending_ensures.insert(Self::expr_key(call), pending);
        }
    }

    fn instantiate_fact(
        &self,
        f: &EnsuresFact,
        args: &[HirExpr<'a, 'bump>],
        base: usize,
    ) -> Option<InstFact> {
        let FactTarget::Param(pidx) = f.target else {
            return None; // return-binder facts are applied by `apply_ret_ensures`
        };
        let HirExpr::Ref { expr: inner, .. } = args.get(base + pidx)? else {
            return None;
        };
        let (target, shift): (&HirExpr<'a, 'bump>, i64) = match &**inner {
            HirExpr::Slice { object, start, .. } => match &**start {
                HirExpr::Number(s, _) => (&**object, *s),
                _ => return None,
            },
            other => (other, 0),
        };
        let (root, mut path) = self.static_field_path(target)?;
        let len = if f.path.is_empty() {
            match self.peek_type(target).kind {
                HirTypeKind::Array(_, l) => Some(l),
                _ => None,
            }
        } else {
            None
        };
        path.extend(f.path.iter().copied());

        let conv = |b: FactBound| -> Option<InstBound> {
            match b {
                FactBound::Const(c) => Some(InstBound::Const(c + shift)),
                FactBound::Binder(n) => (shift == 0).then_some(InstBound::Binder(n)),
                FactBound::Param(j, off) => match args.get(base + j)? {
                    HirExpr::Number(k, _) => Some(InstBound::Const(*k + off + shift)),
                    HirExpr::Ident(v, _) if shift == 0 && off == 0 => Some(InstBound::Var(*v)),
                    _ => None,
                },
                FactBound::ParamLen(j, off) => match args.get(base + j)? {
                    HirExpr::Ref { expr: e2, .. } => match &**e2 {
                        HirExpr::Slice { start, end, .. } => match (&**start, &**end) {
                            (HirExpr::Number(a, _), HirExpr::Number(b, _)) => {
                                Some(InstBound::Const(b - a + off + shift))
                            }
                            _ => None,
                        },
                        other => match self.peek_type(other).kind {
                            HirTypeKind::Array(_, l) => {
                                Some(InstBound::Const(l as i64 + off + shift))
                            }
                            _ => None,
                        },
                    },
                    _ => None,
                },
            }
        };

        Some(InstFact {
            guard: f.guard.clone(),
            root,
            path,
            lo: conv(f.lo)?,
            hi: conv(f.hi)?,
            len,
        })
    }

    /// `match (unsafe { call })` / `match call`: find the call node.
    pub(crate) fn scrutinee_call_expr<'e>(
        e: &'e HirExpr<'a, 'bump>,
    ) -> Option<&'e HirExpr<'a, 'bump>> {
        match e {
            HirExpr::Call { .. } => Some(e),
            HirExpr::Block { body, .. } => match body.last()? {
                HirStmt::Expr(inner) => Self::scrutinee_call_expr(inner),
                _ => None,
            },
            _ => None,
        }
    }

    pub(crate) fn apply_arm_ensures(
        &mut self,
        call: &HirExpr<'a, 'bump>,
        pattern: &HirPattern<'a, 'bump>,
    ) {
        let Some(facts) = self.pending_ensures.get(&Self::expr_key(call)).cloned() else {
            return;
        };
        let ty = self.peek_type(call);
        self.apply_facts_to_arm(facts, pattern, &ty);
    }

    pub(crate) fn apply_nested_arm_ensures(
        &mut self,
        scrutinee: &HirExpr<'a, 'bump>,
        pattern: &HirPattern<'a, 'bump>,
    ) {
        let HirExpr::Ident(n, _) = scrutinee else {
            return;
        };
        let Some(facts) = self.pending_nested.get(n).cloned() else {
            return;
        };
        let ty = self.peek_type(scrutinee);
        self.apply_facts_to_arm(facts, pattern, &ty);
    }

    fn apply_facts_to_arm(
        &mut self,
        facts: Vec<InstFact>,
        pattern: &HirPattern<'a, 'bump>,
        enum_ty: &HirType<'a, 'bump>,
    ) {
        let Some((variant, arm_bindings)) = Self::enum_variant_pattern(pattern, enum_ty) else {
            return;
        };
        for f in facts {
            let Some(g) = &f.guard else { continue };
            if g.variant != variant {
                continue;
            }
            let resolve = |b: InstBound| -> Option<InstBound> {
                match b {
                    InstBound::Binder(n) => match g.bindings.iter().find(|(_, nm)| *nm == n) {
                        Some((field, _)) => {
                            let arm = arm_bindings.iter().find(|(fl, _)| fl == field)?.1;
                            Some(InstBound::Var(arm))
                        }
                        None => Some(InstBound::Binder(n)),
                    },
                    other => Some(other),
                }
            };
            let (Some(lo), Some(hi)) = (resolve(f.lo), resolve(f.hi)) else {
                continue;
            };
            match &g.next {
                None => {
                    if matches!(lo, InstBound::Binder(_)) || matches!(hi, InstBound::Binder(_)) {
                        continue;
                    }
                    self.sym_init.push(SymInit {
                        root: f.root,
                        path: f.path.clone(),
                        lo,
                        hi,
                    });
                }
                Some((via, inner)) => {
                    let Some(&(_, arm)) = arm_bindings.iter().find(|(fl, _)| fl == via) else {
                        continue;
                    };
                    self.pending_nested.entry(arm).or_default().push(InstFact {
                        guard: Some((**inner).clone()),
                        root: f.root,
                        path: f.path.clone(),
                        lo,
                        hi,
                        len: f.len,
                    });
                }
            }
        }
    }

    fn bound_le(a: InstBound, b: InstBound) -> bool {
        match (a, b) {
            (InstBound::Const(x), InstBound::Const(y)) => x <= y,
            (InstBound::Var(x), InstBound::Var(y)) => x == y,
            _ => false,
        }
    }

    pub(crate) fn inst_bound_of(e: &HirExpr<'a, 'bump>) -> Option<InstBound> {
        match e {
            HirExpr::Number(i, _) => Some(InstBound::Const(*i)),
            HirExpr::Ident(n, _) => Some(InstBound::Var(*n)),
            _ => None,
        }
    }

    pub(crate) fn sym_init_covers(
        &self,
        root: StrId,
        path: &[StrId],
        lo: InstBound,
        hi: InstBound,
    ) -> bool {
        self.sym_init.iter().any(|f| {
            f.root == root && f.path == path && Self::bound_le(f.lo, lo) && Self::bound_le(hi, f.hi)
        })
    }

    pub(crate) fn kill_sym_init(&mut self, name: StrId) {
        self.sym_init.retain(|f| {
            f.root != name && f.lo != InstBound::Var(name) && f.hi != InstBound::Var(name)
        });
        self.pending_nested.remove(&name);
    }
}

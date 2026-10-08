use ir::{
    errors::type_error::TypeErrorKind,
    hir::{AssignmentOperator, HirExpr, HirFunc, HirType, RefKind, StrId},
    ir_hasher::{FxHashMap, HashSet},
    nll_cfg::CfgBuilder,
    span::SourceSpan,
};

use crate::{auto_traits::AutoTrait, naming::type_to_string};

use super::*;

#[derive(Clone, Copy)]
pub struct StaticDecl<'a, 'bump> {
    pub ty: HirType<'a, 'bump>,
    pub const_init: bool,
}

#[derive(Default, Clone)]
pub struct StaticSummary {
    pub requires: HashSet<StrId>, // accessed without being (re)initialised first
    pub ensures: HashSet<StrId>,  // definitely INITIALIZED at every exit
    pub kills: HashSet<StrId>,    // possibly UNINITIALIZED at exit
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub(crate) fn register_static(
        &mut self,
        module_idx: usize,
        name: StrId,
        ty: HirType<'a, 'bump>,
        value: &HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    ) {
        self.set_span(span);
        if matches!(ty, HirType::Unknown) {
            self.record(TypeErrorKind::Generic(format!(
                "static `{}` needs an explicit type",
                name
            )));
            return;
        }
        let const_init = match value {
            HirExpr::Uninit { .. } => false,
            v if self.is_const_initializer(module_idx, v) => true,
            _ => {
                self.record(TypeErrorKind::Generic(format!(
                    "initializer of static `{}` must be a compile-time constant; \
                     write `= uninit` and assign it at runtime",
                    name
                )));
                false
            }
        };
        self.module_statics
            .entry(module_idx)
            .or_default()
            .insert(name, StaticDecl { ty, const_init });
    }

    fn is_const_initializer(&self, module_idx: usize, e: &HirExpr<'a, 'bump>) -> bool {
        match e {
            HirExpr::Number(..)
            | HirExpr::Decimal(..)
            | HirExpr::Boolean(..)
            | HirExpr::String(..)
            | HirExpr::Char(..)
            | HirExpr::Null(..)
            | HirExpr::Undefined { .. } => true, // `undefined` = zeroed data
            HirExpr::Tuple(es, _) => es.iter().all(|x| self.is_const_initializer(module_idx, x)),
            HirExpr::ArrayLiteral { elements, .. } => elements
                .iter()
                .all(|x| self.is_const_initializer(module_idx, x)),
            HirExpr::Cast { expr, .. } => self.is_const_initializer(module_idx, expr),
            HirExpr::Binary { left, right, .. } => {
                self.is_const_initializer(module_idx, left)
                    && self.is_const_initializer(module_idx, right)
            }
            HirExpr::Ident(n, _) => self
                .module_consts
                .get(&module_idx)
                .is_some_and(|m| m.contains_key(n)),
            HirExpr::StructInit { args, .. } => args
                .iter()
                .all(|f| self.is_const_initializer(module_idx, &f.value)),
            HirExpr::EnumInit { args, .. } => args
                .iter()
                .all(|v| self.is_const_initializer(module_idx, v)),
            _ => false,
        }
    }

    pub(crate) fn is_static_place(&self, e: &HirExpr<'a, 'bump>) -> bool {
        Self::static_root(e).is_some_and(|r| self.is_static_ident(r))
    }

    pub(crate) fn declare_statics_in_scope(&mut self, ctx: &mut TypeContext<'a, 'bump>) {
        let m = self.context.current_module_idx;
        let mut visible: Vec<(StrId, StaticDecl<'a, 'bump>)> = Vec::new();
        if let Some(own) = self.module_statics.get(&m) {
            visible.extend(own.iter().map(|(k, v)| (*k, *v)));
        }
        if let Some(imps) = self.imports_by_module.get(&m) {
            for (name, real) in imps.named.iter() {
                if let Some(d) = self.module_statics.get(real).and_then(|s| s.get(name)) {
                    visible.push((*name, *d));
                }
            }
        }
        self.cur_statics.clear();
        for (name, d) in visible {
            let sym = self.mint_symbol_id();
            ctx.add_variable(str_id_to_string(name), d.ty, sym);
            self.static_symbols.insert(sym, name);
            self.cur_statics.push(name);
            self.mark_whole_init(name); // entry assumption
        }
        self.fn_static_accessed.clear();
        self.fn_static_assigned.clear();
    }

    pub(crate) fn is_static_ident(&self, name: StrId) -> bool {
        self.context
            .get_variable(&str_id_to_string(name))
            .is_some_and(|(sym, _)| self.static_symbols.contains_key(&sym))
    }

    pub fn is_whole_init(&self, name: StrId) -> bool {
        self.init_state
            .get(&name)
            .map_or(true, |n| n.is_fully_init())
    }

    pub(crate) fn note_static_use(&mut self, name: StrId, mutating: bool) {
        if !self.is_static_ident(name) {
            return;
        }
        self.fn_static_accessed.insert(name);
        for f in self.closure_frames.borrow_mut().iter_mut() {
            match f.static_uses.iter_mut().find(|(n, _)| *n == name) {
                Some((_, m)) => *m |= mutating,
                None => f.static_uses.push((name, mutating)),
            }
        }
    }

    pub(crate) fn note_static_access(&mut self, name: StrId) {
        self.note_static_use(name, false);
    }

    fn plain_assign(op: &AssignmentOperator) -> bool {
        matches!(op, AssignmentOperator::Assign)
    }

    pub(crate) fn static_assignment_pre(
        &mut self,
        target: &HirExpr<'a, 'bump>,
        op: &AssignmentOperator,
    ) {
        if let Some(root) = Self::static_root(target) {
            if self.is_static_ident(root) {
                self.note_static_use(root, true);
            }
        }
        if let HirExpr::Ident(n, _) = target {
            if self.is_static_ident(*n) && Self::plain_assign(op) {
                self.static_whole_target = Some(*n);
                // MIR needs to know whether the old value must be dropped.
                self.static_assign_backfill.insert(
                    Self::expr_key(target),
                    StaticAssignInfo {
                        drop_old: self.is_whole_init(*n),
                    },
                );
            }
        }
    }

    pub(crate) fn static_assignment_post(&mut self, value: &HirExpr<'a, 'bump>) {
        let Some(n) = self.static_whole_target.take() else {
            return;
        };
        if matches!(value, HirExpr::Uninit { .. }) {
            self.mark_whole_uninit(n);
        } else {
            self.mark_whole_init(n);
            self.fn_static_assigned.insert(n);
        }
    }

    pub(crate) fn static_root(expr: &HirExpr<'a, 'bump>) -> Option<StrId> {
        match expr {
            HirExpr::Ident(n, _) => Some(*n),
            HirExpr::FieldAccess { object, .. }
            | HirExpr::Get { object, .. }
            | HirExpr::Index { object, .. }
            | HirExpr::Slice { object, .. } => Self::static_root(object),
            _ => None, // deliberately not through Deref
        }
    }

    /// Returns Some(type) if `expr` borrows from a static (statics have no lifetime constraints).
    pub(crate) fn check_static_ref(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
    ) -> Option<HirType<'a, 'bump>> {
        let root = Self::static_root(expr)?;
        if !self.is_static_ident(root) {
            return None;
        }
        self.set_span(span);
        let mut kind = ref_kind;
        if ref_kind == RefKind::Unique {
            self.record(TypeErrorKind::Generic(format!(
                "cannot take `&mut` of static `{}`; use `&alias` or `&`",
                root
            )));
            kind = RefKind::Alias; // avoid cascading errors
        }
        if kind != RefKind::Shared {
            self.note_static_use(root, true);
        }
        let inner = self.check_expr_as_place(expr); // requires INITIALIZED
        Some(HirType::Ref {
            inner: self.context.bump.alloc_value(inner),
            ref_kind: kind,
            provenance: None,
        })
    }

    /// Call this where spawned-closure captures are checked for Send, with frame.static_uses.
    pub(crate) fn check_static_thread_bounds(
        &mut self,
        uses: &[(StrId, bool)],
        span: SourceSpan<'a>,
    ) {
        self.set_span(span);
        for &(name, mutated) in uses {
            let Some((_, ty)) = self.context.get_variable(&str_id_to_string(name)) else {
                continue;
            };
            if !self.implements_auto(&ty, AutoTrait::Send) {
                self.record(TypeErrorKind::Generic(format!(
                    "static `{}` of type `{}` is used across a thread boundary but is not `Send`",
                    name,
                    type_to_string(&ty)
                )));
            }
            if mutated && !self.implements_auto(&ty, AutoTrait::Sync) {
                self.record(TypeErrorKind::Generic(format!(
                    "static `{}` of type `{}` is mutated or aliased across a thread boundary \
                     but is not `Sync`",
                    name,
                    type_to_string(&ty)
                )));
            }
        }
    }

    pub(crate) fn finish_static_summary(&mut self, func: &HirFunc<'a, 'bump>) {
        let mut s = StaticSummary::default();
        for &st in &self.cur_statics {
            if !self.is_whole_init(st) {
                s.kills.insert(st);
            } else if self.fn_static_assigned.contains(&st) {
                s.ensures.insert(st);
            }
            if self.fn_static_accessed.contains(&st) && !self.fn_static_assigned.contains(&st) {
                s.requires.insert(st);
            }
        }
        self.static_summaries.insert(func.name, s.clone());
        self.static_summaries.insert(func.unmangled_name, s);
    }

    /// Called after a Call expression has been checked.
    pub(crate) fn apply_static_call_effects(
        &mut self,
        callee: &HirExpr<'a, 'bump>,
        span: SourceSpan<'a>,
    ) {
        let HirExpr::Ident(f, _) = callee else { return };
        let Some(sum) = self.static_summaries.get(f).cloned() else {
            return;
        };
        self.set_span(span);
        for &st in &sum.requires {
            if !self.is_whole_init(st) {
                self.record(TypeErrorKind::Generic(format!(
                    "call to `{}` requires static `{}` to be initialized, but it may be uninitialized here",
                    f, st
                )));
            }
            self.fn_static_accessed.insert(st); // transitive
        }
        for &st in &sum.kills {
            self.mark_whole_uninit(st);
        }
        for &st in &sum.ensures {
            self.mark_whole_init(st);
            self.fn_static_assigned.insert(st);
        }
    }

    pub(crate) fn idents_used_in(&mut self, func: &HirFunc<'a, 'bump>) -> HashSet<StrId> {
        let mut out = HashSet::default();
        if let Some(body) = func.body {
            let (cfg, points) = CfgBuilder::new().build(&body);
            self.cfg = cfg;
            self.stmt_points = points.stmt_points;
            self.stmt_after_points = points.stmt_after_points;
            self.point_locals_used = FxHashMap::default();
            self.current_point = self.cfg.entry.unwrap_or_default();
            self.collect_locals_used_stmt(&body);
            for set in self.point_locals_used.values() {
                out.extend(set.iter().copied());
            }
        }
        out
    }
}

#[derive(Clone, Copy)]
pub struct StaticAssignInfo {
    pub drop_old: bool,
}

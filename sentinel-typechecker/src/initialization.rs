use ir::{
    errors::type_error::TypeErrorKind,
    hir::{AssignmentOperator, HirExpr, HirFunc, HirParam, HirStmt, HirType, RefKind, StrId},
    ir_hasher::{FxHashMap, HashSet},
};

use crate::{move_state::MoveState, str_id_to_string, TypeChecker};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingMode {
    ByValue,
    ByRef(RefKind),
}

#[derive(Clone, Copy)]
pub struct WriteFlow {
    /// True if every path that falls through this statement (rather than
    /// returning) has definitely written `*root` by the time it does.
    pub(crate) written: bool,
    /// True if this statement never falls through (every path returns, or
    /// every branch diverges).
    pub(crate) diverges: bool,
    /// True if every early `return` reached from within this statement had
    /// already written `*root` before returning.
    pub(crate) returns_ok: bool,
}

#[derive(Clone, Copy)]
pub enum BareImportKind {
    Struct,
    Enum,
}

impl BareImportKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            BareImportKind::Struct => "struct",
            BareImportKind::Enum => "enum",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum UseLevel {
    Read,
    Alias,
    Mut,
    Move,
}

#[derive(Default)]
pub struct FreeVars {
    order: Vec<StrId>,
    seen: HashSet<StrId>,
}

impl FreeVars {
    pub fn add(&mut self, n: StrId) {
        if self.seen.insert(n) {
            self.order.push(n);
        }
    }
}

pub(crate) struct ModuleImports {
    /// `import foo::bar.Baz;`, Baz becomes usable bare in this module.
    pub(crate) named: FxHashMap<StrId, usize>, // item name -> resolved declaring module_idx
    /// `import foo::bar;`, foo::bar::whatever() becomes usable qualified,
    /// but nothing from it becomes usable bare.
    pub(crate) modules: std::collections::HashSet<usize>,
    pub(crate) module_aliases: FxHashMap<StrId, usize>,
    pub(crate) wildcard: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InitStatus {
    Uninitialized,
    Initialized,
    Maybe,
}

#[derive(Debug, Clone, PartialEq)]
pub enum InitNode {
    Whole(InitStatus),
    Struct(FxHashMap<StrId, InitNode>),
    Array {
        ranges: IntervalSet,
        len: Option<usize>,
    },
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntervalSet {
    // sorted, non-overlapping, non-adjacent ranges [start, end)
    pub ranges: Vec<(i64, i64)>,
}

impl IntervalSet {
    pub fn insert(&mut self, start: i64, end: i64) {
        if start >= end {
            return;
        }
        let mut new_start = start;
        let mut new_end = end;
        let mut result = Vec::with_capacity(self.ranges.len() + 1);
        let mut inserted = false;
        for &(s, e) in &self.ranges {
            if e < new_start {
                result.push((s, e));
            } else if s > new_end {
                if !inserted {
                    result.push((new_start, new_end));
                    inserted = true;
                }
                result.push((s, e));
            } else {
                new_start = new_start.min(s);
                new_end = new_end.max(e);
            }
        }
        if !inserted {
            result.push((new_start, new_end));
        }
        self.ranges = result;
    }

    pub fn contains_range(&self, start: i64, end: i64) -> bool {
        self.ranges.iter().any(|&(s, e)| s <= start && end <= e)
    }

    pub fn covers_full(&self, len: i64) -> bool {
        self.contains_range(0, len)
    }

    /// Intersection, used for CFG-join: a range is only "known initialized"
    /// on the merged path if it was initialized on *both* incoming paths.
    pub fn intersect(&self, other: &IntervalSet) -> IntervalSet {
        let mut result = IntervalSet::default();
        for &(s1, e1) in &self.ranges {
            for &(s2, e2) in &other.ranges {
                let s = s1.max(s2);
                let e = e1.min(e2);
                if s < e {
                    result.insert(s, e);
                }
            }
        }
        result
    }
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn optimistically_mark_mut_target_init(&mut self, expr: &HirExpr<'a, 'bump>) {
        match expr {
            HirExpr::Ident(_, _) | HirExpr::FieldAccess { .. } | HirExpr::Get { .. } => {
                if let Some((root, path)) = self.static_field_path(expr) {
                    self.mark_field_init(root, &path);
                }
            }
            HirExpr::Index { object, index, .. } => {
                if let Some((root, path)) = self.static_field_path(object) {
                    match index {
                        HirExpr::Number(i, _) => {
                            let len = match self.peek_type(object) {
                                HirType::Array(_, l) => Some(l),
                                _ => None,
                            };
                            self.mark_array_range(root, &path, *i, *i + 1, len);
                        }
                        _ => {
                            let node = self
                                .init_state
                                .entry(root)
                                .or_insert(InitNode::Whole(InitStatus::Uninitialized));
                            *Self::node_at_path_mut(node, &path) =
                                InitNode::Whole(InitStatus::Initialized);
                        }
                    }
                }
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                if let Some((root, path)) = self.static_field_path(object) {
                    if let (HirExpr::Number(s, _), HirExpr::Number(e, _)) = (start, end) {
                        let len = match self.peek_type(object) {
                            HirType::Array(_, l) => Some(l),
                            _ => None,
                        };
                        self.mark_array_range(root, &path, *s, *e, len);
                    } else {
                        let node = self
                            .init_state
                            .entry(root)
                            .or_insert(InitNode::Whole(InitStatus::Uninitialized));
                        *Self::node_at_path_mut(node, &path) =
                            InitNode::Whole(InitStatus::Initialized);
                    }
                }
            }
            _ => {}
        }
    }

    pub fn check_expr_suppressed(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        let prev = self.suppress_init_read;
        self.suppress_init_read = true;
        let ty = self.check_expr(expr);
        self.suppress_init_read = prev;
        ty
    }

    pub fn check_read_for_compound_target(
        &mut self,
        target: &HirExpr<'a, 'bump>,
        ty: &HirType<'a, 'bump>,
    ) {
        match target {
            HirExpr::Ident(name, _) => {
                let var_name = str_id_to_string(*name);
                self.check_ident_init_read(*name, &var_name, ty);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let Some((root, mut path)) = self.static_field_path(object) {
                    path.push(*field);
                    let root_str = str_id_to_string(root);
                    self.check_init_read_path(root, &path, &root_str);
                }
            }
            HirExpr::Index { object, index, .. } => {
                if let (Some((root, path)), HirExpr::Number(i, _)) =
                    (self.static_field_path(object), index)
                {
                    if let Some(node) = self.init_state.get(&root).cloned() {
                        let target_node = Self::node_at_path_ref(&node, &path);
                        let covered = match target_node {
                            InitNode::Array { ranges, .. } => ranges.contains_range(*i, *i + 1),
                            InitNode::Whole(InitStatus::Initialized) => true,
                            _ => false,
                        };
                        if !covered {
                            let root_str = str_id_to_string(root);
                            self.record(TypeErrorKind::Generic(format!(
                                "use of uninitialized value `{}[{}]`: a compound assignment reads the current value first",
                                root_str, i
                            )));
                        }
                    }
                }
            }
            _ => {}
        }
    }

    pub fn mark_whole_uninit(&mut self, root: StrId) {
        self.init_state
            .insert(root, InitNode::Whole(InitStatus::Uninitialized));
    }

    pub fn mark_whole_init(&mut self, root: StrId) {
        self.init_state
            .insert(root, InitNode::Whole(InitStatus::Initialized));
    }

    pub fn mark_field_init(&mut self, root: StrId, path: &[StrId]) {
        let node = self
            .init_state
            .entry(root)
            .or_insert(InitNode::Whole(InitStatus::Initialized));
        Self::mark_field_init_node(node, path);
    }

    pub fn mark_field_uninit(&mut self, root: StrId, path: &[StrId]) {
        let node = self
            .init_state
            .entry(root)
            .or_insert(InitNode::Whole(InitStatus::Initialized));
        let target = Self::node_at_path_mut(node, path);
        *target = InitNode::Whole(InitStatus::Uninitialized);
    }

    pub fn mark_field_init_node(node: &mut InitNode, path: &[StrId]) {
        let Some((head, rest)) = path.split_first() else {
            *node = InitNode::Whole(InitStatus::Initialized);
            return;
        };
        let map = match node {
            InitNode::Struct(m) => m,
            InitNode::Whole(InitStatus::Initialized) => return,
            _ => {
                *node = InitNode::Struct(FxHashMap::default());
                let InitNode::Struct(m) = node else {
                    unreachable!()
                };
                m
            }
        };
        let child = map
            .entry(*head)
            .or_insert(InitNode::Whole(InitStatus::Uninitialized));
        Self::mark_field_init_node(child, rest);
    }

    pub fn mark_array_range(
        &mut self,
        root: StrId,
        path: &[StrId],
        start: i64,
        end: i64,
        len: Option<usize>,
    ) {
        let node = self
            .init_state
            .entry(root)
            .or_insert(InitNode::Whole(InitStatus::Uninitialized));
        Self::mark_array_range_node(node, path, start, end, len);
    }

    pub fn mark_array_range_node(
        node: &mut InitNode,
        path: &[StrId],
        start: i64,
        end: i64,
        len: Option<usize>,
    ) {
        if let Some((head, rest)) = path.split_first() {
            let map = match node {
                InitNode::Struct(m) => m,
                _ => {
                    *node = InitNode::Struct(FxHashMap::default());
                    let InitNode::Struct(m) = node else {
                        unreachable!()
                    };
                    m
                }
            };
            let child = map
                .entry(*head)
                .or_insert(InitNode::Whole(InitStatus::Uninitialized));
            return Self::mark_array_range_node(child, rest, start, end, len);
        }
        match node {
            InitNode::Array { ranges, .. } => ranges.insert(start, end),
            InitNode::Whole(InitStatus::Initialized) => {}
            _ => {
                let mut ranges = IntervalSet::default();
                ranges.insert(start, end);
                *node = InitNode::Array { ranges, len };
            }
        }
    }

    pub fn check_init_read_path(&mut self, root: StrId, path: &[StrId], root_str: &str) {
        if self.suppress_init_read {
            return;
        }
        let Some(node) = self.init_state.get(&root) else {
            return;
        };
        match Self::status_at_path(node, path) {
            InitStatus::Initialized => {}
            InitStatus::Uninitialized => self.record(TypeErrorKind::Generic(format!(
                "use of uninitialized value `{}{}`: not assigned since `uninit`",
                root_str,
                path.iter().map(|s| format!(".{}", s)).collect::<String>()
            ))),
            InitStatus::Maybe => self.record(TypeErrorKind::Generic(format!(
                "use of possibly uninitialized value `{}{}`: not initialized on all control-flow paths",
                root_str,
                path.iter().map(|s| format!(".{}", s)).collect::<String>()
            ))),
        }
    }

    pub fn status_at_path(node: &InitNode, path: &[StrId]) -> InitStatus {
        match (node, path.split_first()) {
            (InitNode::Whole(s), None) => s.clone(),
            (InitNode::Whole(InitStatus::Initialized), Some(_)) => InitStatus::Initialized,
            (InitNode::Whole(s), Some(_)) => s.clone(),
            (InitNode::Struct(map), Some((head, rest))) => match map.get(head) {
                Some(child) => Self::status_at_path(child, rest),
                None => InitStatus::Uninitialized, // never touched
            },
            (InitNode::Struct(map), None) => {
                if map
                    .values()
                    .all(|v| matches!(v, InitNode::Whole(InitStatus::Initialized)))
                {
                    InitStatus::Initialized
                } else {
                    InitStatus::Maybe
                }
            }
            (InitNode::Array { .. }, _) => InitStatus::Initialized,
        }
    }

    pub fn join_init_states(
        a: &FxHashMap<StrId, InitNode>,
        b: &FxHashMap<StrId, InitNode>,
    ) -> FxHashMap<StrId, InitNode> {
        let mut result = FxHashMap::default();
        for key in a.keys().chain(b.keys()).copied().collect::<HashSet<_>>() {
            let na = a
                .get(&key)
                .cloned()
                .unwrap_or(InitNode::Whole(InitStatus::Initialized));
            let nb = b
                .get(&key)
                .cloned()
                .unwrap_or(InitNode::Whole(InitStatus::Initialized));
            result.insert(key, Self::join_nodes(na, nb));
        }
        result
    }

    pub fn join_nodes(a: InitNode, b: InitNode) -> InitNode {
        match (a, b) {
            (InitNode::Whole(sa), InitNode::Whole(sb)) => InitNode::Whole(match (sa, sb) {
                (InitStatus::Initialized, InitStatus::Initialized) => InitStatus::Initialized,
                (InitStatus::Uninitialized, InitStatus::Uninitialized) => InitStatus::Uninitialized,
                _ => InitStatus::Maybe,
            }),
            (InitNode::Array { ranges: ra, len }, InitNode::Array { ranges: rb, .. }) => {
                InitNode::Array {
                    ranges: ra.intersect(&rb),
                    len,
                }
            }
            (InitNode::Array { ranges, len }, InitNode::Whole(InitStatus::Initialized))
            | (InitNode::Whole(InitStatus::Initialized), InitNode::Array { ranges, len }) => {
                InitNode::Array { ranges, len }
            }
            (InitNode::Struct(ma), InitNode::Struct(mb)) => {
                let mut out = FxHashMap::default();
                for key in ma.keys().chain(mb.keys()).copied().collect::<HashSet<_>>() {
                    let fa = ma
                        .get(&key)
                        .cloned()
                        .unwrap_or(InitNode::Whole(InitStatus::Uninitialized));
                    let fb = mb
                        .get(&key)
                        .cloned()
                        .unwrap_or(InitNode::Whole(InitStatus::Uninitialized));
                    out.insert(key, Self::join_nodes(fa, fb));
                }
                InitNode::Struct(out)
            }
            (InitNode::Struct(ma), InitNode::Whole(InitStatus::Initialized))
            | (InitNode::Whole(InitStatus::Initialized), InitNode::Struct(ma)) => {
                InitNode::Struct(ma)
            }
            _ => InitNode::Whole(InitStatus::Maybe),
        }
    }

    pub fn converge_loop_states(
        &mut self,
        body: &HirStmt<'a, 'bump>,
        move_entry: MoveState,
        init_entry: FxHashMap<StrId, InitNode>,
    ) -> (MoveState, FxHashMap<StrId, InitNode>) {
        let saved_move_state = self.move_state.clone();
        let saved_init_state = self.init_state.clone();
        let saved_context = self.context.clone();

        let mut converged_move = move_entry;
        let mut converged_init = init_entry;

        self.with_suppressed_errors(|this| loop {
            this.move_state = converged_move.clone();
            this.init_state = converged_init.clone();
            this.check_stmt(body);
            let next_move = MoveState::join(&converged_move, &this.move_state);
            let next_init = Self::join_init_states(&converged_init, &this.init_state);

            let stable = converged_move.is_superset_of(&next_move) && converged_init == next_init;
            converged_move = next_move;
            converged_init = next_init;
            if stable {
                break;
            }
        });

        self.move_state = saved_move_state;
        self.init_state = saved_init_state;
        self.context = saved_context;
        (converged_move, converged_init)
    }

    pub fn check_ident_init_read(&mut self, name: StrId, var_name: &str, ty: &HirType<'a, 'bump>) {
        if self.suppress_init_read {
            return;
        }
        let Some(node) = self.init_state.get(&name).cloned() else {
            return;
        };
        let status = match ty {
            HirType::Struct { name: sname, .. } => self.status_for_whole_struct(&node, *sname),
            _ => Self::status_at_path(&node, &[]),
        };
        match status {
            InitStatus::Initialized => {}
            InitStatus::Uninitialized => self.record(TypeErrorKind::Generic(format!(
                "use of uninitialized value `{}`",
                var_name
            ))),
            InitStatus::Maybe => self.record(TypeErrorKind::Generic(format!(
                "use of possibly uninitialized value `{}`",
                var_name
            ))),
        }
    }

    pub fn check_slice_range_init(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        start: &HirExpr<'a, 'bump>,
        end: &HirExpr<'a, 'bump>,
    ) {
        if self.suppress_init_read {
            return;
        }
        let Some((root, path)) = self.static_field_path(object) else {
            return;
        };
        let Some(node) = self.init_state.get(&root).cloned() else {
            return;
        };
        let target = Self::node_at_path_ref(&node, &path);
        let root_str = str_id_to_string(root);

        match (start, end) {
            (HirExpr::Number(s, _), HirExpr::Number(e, _)) => {
                let covered = match target {
                    InitNode::Array { ranges, .. } => ranges.contains_range(*s, *e),
                    InitNode::Whole(InitStatus::Initialized) => true,
                    _ => false,
                };
                if !covered {
                    self.record(TypeErrorKind::Generic(format!(
                        "use of uninitialized value `{}[{}..{}]`: not every element in this range has been assigned since `uninit`",
                        root_str, s, e
                    )));
                }
            }
            _ => {
                let whole_ok = match target {
                    InitNode::Whole(InitStatus::Initialized) => true,
                    InitNode::Array {
                        ranges,
                        len: Some(l),
                    } => ranges.covers_full(*l as i64),
                    _ => false,
                };
                if !whole_ok {
                    self.record(TypeErrorKind::Generic(format!(
                        "slicing `{}` with a non-constant bound requires the whole array to be initialized \
                         (the compiler can't prove which elements are covered)",
                        root_str
                    )));
                }
            }
        }
    }

    pub fn analyze_definite_writes(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<bool> {
        if let Some(t) = self.write_templates.get(&func.name) {
            return t.clone();
        }

        self.write_templates.insert(func.name, Vec::new());
        let templates = Self::build_write_templates(func);
        self.write_templates.insert(func.name, templates.clone());
        templates
    }

    pub fn build_write_templates(func: &HirFunc<'a, 'bump>) -> Vec<bool> {
        let Some(params) = func.params else {
            return Vec::new();
        };
        let mut param_names: Vec<StrId> = Vec::new();
        for p in params.iter() {
            if let HirParam::Normal { name, .. } = p {
                param_names.push(*name);
            }
        }
        let Some(body) = func.body else {
            return vec![false; param_names.len()];
        };
        param_names
            .iter()
            .map(|&name| {
                let flow = Self::analyze_write_stmt(&body, name, false);
                flow.returns_ok && (flow.diverges || flow.written)
            })
            .collect()
    }

    pub fn analyze_write_stmt(
        stmt: &HirStmt<'a, 'bump>,
        root: StrId,
        written_before: bool,
    ) -> WriteFlow {
        match stmt {
            HirStmt::Expr(e) => {
                if Self::expr_is_full_deref_write(e, root) {
                    WriteFlow {
                        written: true,
                        diverges: false,
                        returns_ok: true,
                    }
                } else {
                    WriteFlow {
                        written: written_before,
                        diverges: false,
                        returns_ok: true,
                    }
                }
            }
            HirStmt::Block { body, span: _ } => {
                Self::analyze_write_block(body, root, written_before)
            }
            HirStmt::If {
                then_block,
                else_block,
                ..
            } => {
                let then_flow = Self::analyze_write_block(then_block, root, written_before);
                let else_flow = match else_block {
                    Some(else_stmt) => Self::analyze_write_stmt(else_stmt, root, written_before),
                    None => WriteFlow {
                        written: written_before,
                        diverges: false,
                        returns_ok: true,
                    },
                };
                let returns_ok = then_flow.returns_ok && else_flow.returns_ok;
                match (then_flow.diverges, else_flow.diverges) {
                    (true, true) => WriteFlow {
                        written: true,
                        diverges: true,
                        returns_ok,
                    },
                    (true, false) => WriteFlow {
                        written: else_flow.written,
                        diverges: false,
                        returns_ok,
                    },
                    (false, true) => WriteFlow {
                        written: then_flow.written,
                        diverges: false,
                        returns_ok,
                    },
                    (false, false) => WriteFlow {
                        written: then_flow.written && else_flow.written,
                        diverges: false,
                        returns_ok,
                    },
                }
            }
            HirStmt::Match { arms, .. } => {
                if arms.is_empty() {
                    return WriteFlow {
                        written: written_before,
                        diverges: false,
                        returns_ok: true,
                    };
                }
                let mut returns_ok = true;
                let mut all_diverge = true;
                let mut all_written = true;
                for arm in arms.iter() {
                    let flow = Self::analyze_write_stmt(arm.body, root, written_before);
                    returns_ok &= flow.returns_ok;
                    if flow.diverges {
                        continue;
                    }
                    all_diverge = false;
                    all_written &= flow.written;
                }
                if all_diverge {
                    WriteFlow {
                        written: true,
                        diverges: true,
                        returns_ok,
                    }
                } else {
                    WriteFlow {
                        written: all_written,
                        diverges: false,
                        returns_ok,
                    }
                }
            }
            HirStmt::Return(_, _span) => WriteFlow {
                written: written_before,
                diverges: true,
                returns_ok: written_before,
            },
            HirStmt::Break(..) | HirStmt::Continue(_) => WriteFlow {
                written: written_before,
                diverges: true,
                returns_ok: true,
            },
            HirStmt::While { .. } | HirStmt::For { .. } => WriteFlow {
                written: written_before,
                diverges: false,
                returns_ok: true,
            },
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                Self::analyze_write_stmt(body, root, written_before)
            }
            HirStmt::Let { .. }
            | HirStmt::Const(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => WriteFlow {
                written: written_before,
                diverges: false,
                returns_ok: true,
            },
        }
    }

    pub fn analyze_write_block(
        body: &[HirStmt<'a, 'bump>],
        root: StrId,
        written_before: bool,
    ) -> WriteFlow {
        let mut written = written_before;
        let mut diverges = false;
        let mut returns_ok = true;
        for s in body.iter() {
            if diverges {
                break;
            }
            let flow = Self::analyze_write_stmt(s, root, written);
            returns_ok &= flow.returns_ok;
            written = flow.written;
            diverges = flow.diverges;
        }
        WriteFlow {
            written,
            diverges,
            returns_ok,
        }
    }

    pub fn expr_is_full_deref_write(expr: &HirExpr<'a, 'bump>, root: StrId) -> bool {
        match expr {
            HirExpr::Assignment { target, op, .. } => {
                matches!(op, AssignmentOperator::Assign) && Self::is_deref_of(target, root)
            }
            _ => false,
        }
    }

    pub fn is_deref_of(expr: &HirExpr<'a, 'bump>, root: StrId) -> bool {
        match expr {
            HirExpr::Deref { expr: inner, .. } => Self::is_ident(inner, root),
            _ => false,
        }
    }

    pub fn is_ident(expr: &HirExpr<'a, 'bump>, root: StrId) -> bool {
        match expr {
            HirExpr::Ident(name, _) => *name == root,
            _ => false,
        }
    }

    pub fn node_at_path_mut<'n>(node: &'n mut InitNode, path: &[StrId]) -> &'n mut InitNode {
        let Some((head, rest)) = path.split_first() else {
            return node;
        };
        let map = match node {
            InitNode::Struct(m) => m,
            _ => {
                *node = InitNode::Struct(FxHashMap::default());
                let InitNode::Struct(m) = node else {
                    unreachable!()
                };
                m
            }
        };
        let child = map
            .entry(*head)
            .or_insert(InitNode::Whole(InitStatus::Uninitialized));
        Self::node_at_path_mut(child, rest)
    }

    pub fn status_for_whole_struct(&self, node: &InitNode, struct_name: StrId) -> InitStatus {
        let InitNode::Struct(map) = node else {
            return match node {
                InitNode::Whole(s) => s.clone(),
                _ => InitStatus::Initialized,
            };
        };
        let struct_name_str = str_id_to_string(struct_name);
        let Some(def) = self.context.get_struct(&struct_name_str) else {
            return InitStatus::Maybe;
        };
        let mut any_uninit = false;
        let mut any_init = false;
        for field in def.fields.iter() {
            let status = map
                .get(&field.name)
                .map(|n| Self::status_at_path(n, &[]))
                .unwrap_or(InitStatus::Uninitialized);
            match status {
                InitStatus::Initialized => any_init = true,
                _ => any_uninit = true,
            }
        }
        if any_uninit && any_init {
            InitStatus::Maybe
        } else if any_uninit {
            InitStatus::Uninitialized
        } else {
            InitStatus::Initialized
        }
    }

    pub fn node_at_path_ref<'n>(node: &'n InitNode, path: &[StrId]) -> &'n InitNode {
        let Some((head, rest)) = path.split_first() else {
            return node;
        };
        match node {
            InitNode::Struct(m) => match m.get(head) {
                Some(child) => Self::node_at_path_ref(child, rest),
                None => node,
            },
            _ => node,
        }
    }
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::move_state::MoveState;
use crate::type_context::TypeContext;
use codex_dependency_graph::DepGraph;
use ir::analysis_context::CopyAnalysisCtx;
use ir::ast::{FuncSafety, MutabilityState};
use ir::auto_imports::AutoImportRegistry;
use ir::borrow_checker::{
    BorrowChecker, BorrowError, BorrowKind, Bound, IndexContainer, IndexTemplate, Interval, LoanId,
    MemoryRelation, PlaceId, ReadTemplate, RefTemplate, TemplateBase, TemplateProjection,
};
use ir::errors::type_error::{TypeCheckResult, TypeError, TypeErrorKind};
use ir::hir::{
    AssignmentOperator, EffectIndexKey, Hir, HirEffectAccess, HirEffectSegment,
    HirErrorHandlerPattern, HirExpr, HirFunc, HirMatchArm, HirModule, HirParam, HirPattern,
    HirStmt, HirType, InterpolationPart, IntrinsicKind, Operator, ProvenanceAnnotation,
    ProvenancePathSegment, ProvenanceRoot, RefKind, StrId, ThisPassingKind, Visibility,
};
use ir::ir_hasher::{FxHashBuilder, FxHashMap, HashSet};
use ir::nll_cfg::{Cfg, CfgBuilder, PointId};
use ir::span::SourceSpan;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalSymbolId(u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolId {
    /// let-binding, parameter, or `this`, identified by mint order, since
    /// name alone isn't unique across scopes.
    Local(LocalSymbolId),
    /// struct field, identified directly by (declaring struct, field
    /// name), already globally unique, no minting needed.
    Field {
        struct_name: StrId,
        field_name: StrId,
    },
    /// top-level declaration, same coordinate DepGraph already uses.
    /// Not populated yet, reserved for when function/struct/enum *name*
    /// occurrences get recorded (go-to-def on the declaration side).
    Item {
        module_idx: usize,
        item_idx: usize,
        tag: &'static str,
    },
    Method {
        module_idx: usize,
        item_idx: usize,
        method_idx: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingMode {
    ByValue,
    ByRef(RefKind),
}

#[derive(Clone, Copy)]
struct WriteFlow {
    /// True if every path that falls through this statement (rather than
    /// returning) has definitely written `*root` by the time it does.
    written: bool,
    /// True if this statement never falls through (every path returns, or
    /// every branch diverges).
    diverges: bool,
    /// True if every early `return` reached from within this statement had
    /// already written `*root` before returning.
    returns_ok: bool,
}

#[derive(Clone)]
enum UsedIndex {
    Range(i64, i64),          // from a literal index or a slice
    Place(StrId, Vec<StrId>), // from a stable field-path index
}

type EffectUsage = (Vec<StrId>, Option<UsedIndex>);

#[derive(Clone, Copy)]
enum BareImportKind {
    Struct,
    Enum,
}

impl BareImportKind {
    fn as_str(&self) -> &'static str {
        match self {
            BareImportKind::Struct => "struct",
            BareImportKind::Enum => "enum",
        }
    }
}

struct ModuleImports {
    /// `import foo::bar.Baz;`, Baz becomes usable bare in this module.
    named: FxHashMap<StrId, usize>, // item name -> resolved declaring module_idx
    /// `import foo::bar;`, foo::bar::whatever() becomes usable qualified,
    /// but nothing from it becomes usable bare.
    modules: std::collections::HashSet<usize>,
    module_aliases: FxHashMap<StrId, usize>,
    wildcard: Vec<usize>,
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

const SLICE_PRIMITIVES: &[&str] = &["write_uninit", "write_uninit_all", "get_unchecked"];

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct IntervalSet {
    // sorted, non-overlapping, non-adjacent ranges [start, end)
    ranges: Vec<(i64, i64)>,
}

impl IntervalSet {
    fn insert(&mut self, start: i64, end: i64) {
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

    fn contains_range(&self, start: i64, end: i64) -> bool {
        self.ranges.iter().any(|&(s, e)| s <= start && end <= e)
    }

    fn covers_full(&self, len: i64) -> bool {
        self.contains_range(0, len)
    }

    /// Intersection, used for CFG-join: a range is only "known initialized"
    /// on the merged path if it was initialized on *both* incoming paths.
    fn intersect(&self, other: &IntervalSet) -> IntervalSet {
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

pub struct TypeChecker<'a, 'bump> {
    context: TypeContext<'a, 'bump>,
    errors: Vec<TypeError<'a>>,
    current_span: SourceSpan<'a>,
    copy_analysis: Rc<RefCell<CopyAnalysisCtx<'a, 'bump>>>,
    move_state: MoveState,
    borrow_checker: BorrowChecker,
    this_id: StrId,
    suppress_errors: bool,
    ref_templates: FxHashMap<StrId, RefTemplate>,
    read_templates: FxHashMap<StrId, Vec<ReadTemplate>>,
    write_templates: FxHashMap<StrId, Vec<bool>>,
    next_symbol_id: u32,
    occurrences: Vec<(
        SourceSpan<'a>,
        StrId,
        HirType<'a, 'bump>,
        usize,
        SymbolId,
        bool,
    )>,
    imports_by_module: FxHashMap<usize, ModuleImports>,
    functions_by_module: FxHashMap<usize, HashSet<StrId>>,
    structs_by_module: FxHashMap<usize, HashSet<StrId>>,
    enums_by_module: FxHashMap<usize, HashSet<StrId>>,
    generic_instance_args: FxHashMap<usize, Vec<HirType<'a, 'bump>>>,
    loan_owners: FxHashMap<LoanId, StrId>,
    local_provenance_place: FxHashMap<StrId, PlaceId>,
    call_loans: FxHashMap<usize, LoanId>,
    next_opaque_id: u32,
    unsafe_depth: usize,
    auto_imports: Rc<RefCell<AutoImportRegistry>>,
    cfg: Cfg,
    stmt_points: FxHashMap<usize, PointId>,
    stmt_after_points: FxHashMap<usize, PointId>,
    point_locals_used: FxHashMap<PointId, HashSet<StrId>>,
    current_point: PointId,
    undefined_backfill: FxHashMap<usize, HirType<'a, 'bump>>,
    uninit_backfill: FxHashMap<usize, HirType<'a, 'bump>>,
    init_state: FxHashMap<StrId, InitNode>,
    suppress_init_read: bool,
    local_ref_kind: FxHashMap<StrId, RefKind>,
    binding_mode_backfill: FxHashMap<usize, BindingMode>,
    non_null_state: FxHashMap<StrId, HashSet<Vec<StrId>>>,
    in_place_context: bool,
    skip_slice_init_check: bool,
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn new(
        dep_graph: &'a RefCell<DepGraph>,
        bump: &'bump GrowableBump<'bump>,
        copy_analysis: Rc<RefCell<CopyAnalysisCtx<'a, 'bump>>>,
        string_pool: Arc<StringPool>,
        auto_imports: Rc<RefCell<AutoImportRegistry>>,
    ) -> Self {
        Self {
            this_id: StrId::from_static("this"),
            context: TypeContext::new(dep_graph, bump, string_pool),
            errors: Vec::new(),
            current_span: SourceSpan::default(),
            copy_analysis,
            move_state: MoveState::new(),
            borrow_checker: BorrowChecker::new(),
            suppress_errors: false,
            ref_templates: FxHashMap::default(),
            read_templates: FxHashMap::default(),
            next_symbol_id: 0,
            occurrences: Vec::new(),
            functions_by_module: FxHashMap::default(),
            write_templates: FxHashMap::default(),
            undefined_backfill: FxHashMap::default(),
            imports_by_module: FxHashMap::default(),
            enums_by_module: FxHashMap::default(),
            structs_by_module: FxHashMap::default(),
            generic_instance_args: FxHashMap::default(),
            loan_owners: FxHashMap::default(),
            local_provenance_place: FxHashMap::default(),
            call_loans: FxHashMap::default(),
            next_opaque_id: 0,
            unsafe_depth: 0,
            auto_imports,
            cfg: Cfg::default(),
            stmt_points: HashMap::default(),
            stmt_after_points: HashMap::default(),
            point_locals_used: HashMap::default(),
            current_point: PointId::default(),
            uninit_backfill: FxHashMap::default(),
            init_state: FxHashMap::default(),
            local_ref_kind: FxHashMap::default(),
            binding_mode_backfill: FxHashMap::default(),
            non_null_state: FxHashMap::default(),
            suppress_init_read: false,
            in_place_context: false,
            skip_slice_init_check: false,
        }
    }

    fn optimistically_mark_mut_target_init(&mut self, expr: &HirExpr<'a, 'bump>) {
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

    fn check_expr_suppressed(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        let prev = self.suppress_init_read;
        self.suppress_init_read = true;
        let ty = self.check_expr(expr);
        self.suppress_init_read = prev;
        ty
    }

    fn check_read_for_compound_target(
        &mut self,
        target: &HirExpr<'a, 'bump>,
        ty: &HirType<'a, 'bump>,
    ) {
        match target {
            HirExpr::Ident(name, _) => {
                let var_name = self.str_id_to_string(*name);
                self.check_ident_init_read(*name, &var_name, ty);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let Some((root, mut path)) = self.static_field_path(object) {
                    path.push(*field);
                    let root_str = self.str_id_to_string(root);
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
                            let root_str = self.str_id_to_string(root);
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

    fn mark_whole_uninit(&mut self, root: StrId) {
        self.init_state
            .insert(root, InitNode::Whole(InitStatus::Uninitialized));
    }

    fn mark_whole_init(&mut self, root: StrId) {
        self.init_state
            .insert(root, InitNode::Whole(InitStatus::Initialized));
    }

    fn mark_field_init(&mut self, root: StrId, path: &[StrId]) {
        let node = self
            .init_state
            .entry(root)
            .or_insert(InitNode::Whole(InitStatus::Initialized));
        Self::mark_field_init_node(node, path);
    }

    fn mark_non_null(&mut self, root: StrId, path: &[StrId]) {
        self.non_null_state
            .entry(root)
            .or_default()
            .insert(path.to_vec());
    }

    /// Clears exact-path and every path *below* it (assigning `this.tail = x`
    /// invalidates anything previously proven about `this.tail.next`, etc.)
    fn clear_non_null(&mut self, root: StrId, path: &[StrId]) {
        if let Some(set) = self.non_null_state.get_mut(&root) {
            set.retain(|p| !(p.len() >= path.len() && p[..path.len()] == *path));
        }
    }

    fn is_non_null(&self, root: StrId, path: &[StrId]) -> bool {
        self.non_null_state
            .get(&root)
            .is_some_and(|set| set.contains(path))
    }

    fn join_non_null_states(
        a: &FxHashMap<StrId, HashSet<Vec<StrId>>>,
        b: &FxHashMap<StrId, HashSet<Vec<StrId>>>,
    ) -> FxHashMap<StrId, HashSet<Vec<StrId>>> {
        let mut result = FxHashMap::default();
        for (root, a_paths) in a {
            if let Some(b_paths) = b.get(root) {
                let intersected: HashSet<Vec<StrId>> =
                    a_paths.intersection(b_paths).cloned().collect();
                if !intersected.is_empty() {
                    result.insert(*root, intersected);
                }
            }
        }
        result
    }

    fn condition_to_non_null_fact(
        &self,
        cond: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<StrId>, bool)> {
        let HirExpr::Comparison {
            left, op, right, ..
        } = cond
        else {
            return None;
        };
        match op {
            // `x != null` -> non-null holds in the *true* branch.
            Operator::NotEquals
                if matches!(right, HirExpr::Null(_)) || matches!(left, HirExpr::Null(_)) =>
            {
                let non_null_side = if matches!(right, HirExpr::Null(_)) {
                    left
                } else {
                    right
                };
                let (root, path) = self.static_field_path(non_null_side)?;
                Some((root, path, true))
            }
            // `x == null` -> non-null holds in the *false* (else) branch.
            Operator::Equals
                if matches!(right, HirExpr::Null(_)) || matches!(left, HirExpr::Null(_)) =>
            {
                let non_null_side = if matches!(right, HirExpr::Null(_)) {
                    left
                } else {
                    right
                };
                let (root, path) = self.static_field_path(non_null_side)?;
                Some((root, path, false))
            }
            // `x == 5` (nullable-equality) -> non-null holds in the *true* branch;
            // `x != 5` establishes nothing (`x == null` also satisfies `!= 5`).
            Operator::Equals => {
                let non_null_side = match (left, right) {
                    (_e, HirExpr::Null(_)) | (HirExpr::Null(_), _e) => return None, // handled above
                    (e, _other) if self.static_field_path(e).is_some() => e,
                    _ => return None,
                };
                let (root, path) = self.static_field_path(non_null_side)?;
                Some((root, path, true))
            }
            _ => None,
        }
    }

    fn mark_field_uninit(&mut self, root: StrId, path: &[StrId]) {
        let node = self
            .init_state
            .entry(root)
            .or_insert(InitNode::Whole(InitStatus::Initialized));
        let target = Self::node_at_path_mut(node, path);
        *target = InitNode::Whole(InitStatus::Uninitialized);
    }

    fn mark_field_init_node(node: &mut InitNode, path: &[StrId]) {
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

    fn mark_array_range(
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

    fn mark_array_range_node(
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

    fn static_field_path(&self, expr: &HirExpr<'a, 'bump>) -> Option<(StrId, Vec<StrId>)> {
        match expr {
            HirExpr::Ident(name, _) => Some((*name, Vec::new())),
            HirExpr::This { .. } => Some((self.this_id, Vec::new())),
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let (root, mut path) = self.static_field_path(object)?;
                path.push(*field);
                Some((root, path))
            }
            _ => None,
        }
    }

    fn effect_place_of(
        &self,
        expr: &HirExpr<'a, 'bump>,
    ) -> Option<(StrId, Vec<StrId>, Option<UsedIndex>)> {
        match expr {
            HirExpr::Index { object, index, .. } => {
                let (root, path) = self.static_field_path(object)?;
                let used = match index {
                    HirExpr::Number(i, _) => Some(UsedIndex::Range(*i, *i + 1)),
                    other => self
                        .static_field_path(other)
                        .map(|(r, p)| UsedIndex::Place(r, p)),
                };
                Some((root, path, used))
            }
            _ => self.static_field_path(expr).map(|(r, p)| (r, p, None)),
        }
    }

    fn collect_root_accesses_expr(
        &self,
        expr: &HirExpr<'a, 'bump>,
        root: StrId,
        writes: &mut Vec<EffectUsage>,
        reads: &mut Vec<EffectUsage>,
    ) {
        match expr {
            HirExpr::Assignment {
                target, op, value, ..
            } => {
                self.collect_root_accesses_expr(value, root, writes, reads);
                if let Some((r, path, range)) = self.effect_place_of(target) {
                    if r == root {
                        if !matches!(op, AssignmentOperator::Assign) {
                            writes.push((path.clone(), range.clone()));
                            reads.push((path, range)); // compound assign reads-then-writes
                        } else {
                            writes.push((path.clone(), range));
                        }
                        return;
                    }
                }
                self.collect_root_accesses_expr(target, root, writes, reads);
            }
            HirExpr::Ref {
                expr: inner,
                ref_kind,
                ..
            } => {
                if let Some((r, path, range)) = self.effect_place_of(inner) {
                    if r == root {
                        if *ref_kind != RefKind::Shared {
                            writes.push((path.clone(), range.clone()));
                            reads.push((path, range));
                        } else {
                            reads.push((path, range));
                        }
                        return;
                    }
                }
                self.collect_root_accesses_expr(inner, root, writes, reads);
            }
            HirExpr::Ident(name, _) if *name == root => reads.push((Vec::new(), None)),
            HirExpr::This { .. } if root == self.this_id => reads.push((Vec::new(), None)),
            HirExpr::Ident(_, _) | HirExpr::This { .. } => {}
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let Some((r, mut path)) = self.static_field_path(object) {
                    if r == root {
                        path.push(*field);
                        reads.push((path, None));
                        return;
                    }
                }
                self.collect_root_accesses_expr(object, root, writes, reads);
            }
            HirExpr::Index { object, index, .. } => {
                if let Some((r, path, range)) = self.effect_place_of(expr) {
                    if r == root {
                        reads.push((path, range));
                        self.collect_root_accesses_expr(index, root, writes, reads);
                        return;
                    }
                }
                self.collect_root_accesses_expr(object, root, writes, reads);
                self.collect_root_accesses_expr(index, root, writes, reads);
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                if let Some((r, path)) = self.static_field_path(object) {
                    if r == root {
                        reads.push((path, None));
                        self.collect_root_accesses_expr(start, root, writes, reads);
                        self.collect_root_accesses_expr(end, root, writes, reads);
                        return;
                    }
                }
                self.collect_root_accesses_expr(object, root, writes, reads);
                self.collect_root_accesses_expr(start, root, writes, reads);
                self.collect_root_accesses_expr(end, root, writes, reads);
            }
            HirExpr::Deref { expr: inner, .. } => {
                self.collect_root_accesses_expr(inner, root, writes, reads);
            }
            HirExpr::Range { start, end, .. } => {
                self.collect_root_accesses_expr(start, root, writes, reads);
                self.collect_root_accesses_expr(end, root, writes, reads);
            }
            HirExpr::Tuple(exprs, _)
            | HirExpr::ArrayLiteral {
                elements: exprs, ..
            }
            | HirExpr::ExprList { list: exprs, .. } => {
                for e in exprs.iter() {
                    self.collect_root_accesses_expr(e, root, writes, reads);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                self.collect_root_accesses_expr(left, root, writes, reads);
                self.collect_root_accesses_expr(right, root, writes, reads);
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                self.collect_root_accesses_expr(callee, root, writes, reads);
                for a in args.iter() {
                    self.collect_root_accesses_expr(a, root, writes, reads);
                }
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    self.collect_root_accesses_expr(&f.value, root, writes, reads);
                }
            }
            HirExpr::EnumInit { args, .. } => {
                for a in args.iter() {
                    self.collect_root_accesses_expr(a, root, writes, reads);
                }
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        self.collect_root_accesses_expr(e, root, writes, reads);
                    }
                }
            }
            HirExpr::Cast { expr: inner, .. } => {
                self.collect_root_accesses_expr(inner, root, writes, reads);
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    self.collect_root_accesses_expr(a, root, writes, reads);
                }
            }
            HirExpr::If { if_stmt, .. } => {
                self.collect_root_accesses_stmt(if_stmt, root, writes, reads);
            }
            HirExpr::Match {
                expr: scrutinee,
                arms,
                ..
            } => {
                self.collect_root_accesses_expr(scrutinee, root, writes, reads);
                for arm in arms.iter() {
                    if let Some(guard) = arm.guard {
                        self.collect_root_accesses_expr(guard, root, writes, reads);
                    }
                    self.collect_root_accesses_stmt(arm.body, root, writes, reads);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    self.collect_root_accesses_stmt(s, root, writes, reads);
                }
            }
            HirExpr::Lambda { params, body, .. } => {
                let shadowed = params.iter().any(|p| p.name == root);
                if !shadowed {
                    self.collect_root_accesses_stmt(body, root, writes, reads);
                }
            }
            HirExpr::Null(_)
            | HirExpr::Number(_, _)
            | HirExpr::Char(_, _)
            | HirExpr::String(_, _)
            | HirExpr::Boolean(_, _)
            | HirExpr::Decimal(_, _)
            | HirExpr::Undefined { .. }
            | HirExpr::Uninit { .. }
            | HirExpr::GenericIdent(..)
            | HirExpr::ModuleAccess(_)
            | HirExpr::UnknownIntrinsic { .. } => {}
        }
    }

    fn collect_root_accesses_stmt(
        &self,
        stmt: &HirStmt<'a, 'bump>,
        root: StrId,
        writes: &mut Vec<EffectUsage>,
        reads: &mut Vec<EffectUsage>,
    ) {
        match stmt {
            HirStmt::Let {
                value,
                else_block,
                catch_pattern,
                ..
            } => {
                self.collect_root_accesses_expr(value, root, writes, reads);
                if let Some(b) = else_block {
                    self.collect_root_accesses_stmt(b, root, writes, reads);
                }
                if let Some(pattern) = catch_pattern {
                    match pattern {
                        HirErrorHandlerPattern::Single { body, .. } => {
                            for s in body.iter() {
                                self.collect_root_accesses_stmt(s, root, writes, reads);
                            }
                        }
                        HirErrorHandlerPattern::Multiple { branches } => {
                            for branch in branches.iter() {
                                for s in branch.body.iter() {
                                    self.collect_root_accesses_stmt(s, root, writes, reads);
                                }
                            }
                        }
                    }
                }
            }
            HirStmt::Const(c) => self.collect_root_accesses_expr(&c.value, root, writes, reads),
            HirStmt::Return(Some(e), _span) | HirStmt::Break(Some(e), _span) => {
                self.collect_root_accesses_expr(e, root, writes, reads)
            }
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => {}
            HirStmt::Expr(e) => self.collect_root_accesses_expr(e, root, writes, reads),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                self.collect_root_accesses_expr(cond, root, writes, reads);
                for s in then_block.iter() {
                    self.collect_root_accesses_stmt(s, root, writes, reads);
                }
                if let Some(e) = else_block {
                    self.collect_root_accesses_stmt(e, root, writes, reads);
                }
            }
            HirStmt::While { cond, body } => {
                self.collect_root_accesses_expr(cond, root, writes, reads);
                self.collect_root_accesses_stmt(body, root, writes, reads);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    self.collect_root_accesses_stmt(i, root, writes, reads);
                }
                if let Some(c) = condition {
                    self.collect_root_accesses_expr(c, root, writes, reads);
                }
                if let Some(inc) = increment {
                    self.collect_root_accesses_expr(inc, root, writes, reads);
                }
                self.collect_root_accesses_stmt(body, root, writes, reads);
            }
            HirStmt::Block { body, span: _ } => {
                for s in body.iter() {
                    self.collect_root_accesses_stmt(s, root, writes, reads);
                }
            }
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                self.collect_root_accesses_expr(expr, root, writes, reads);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        self.collect_root_accesses_expr(g, root, writes, reads);
                    }
                    self.collect_root_accesses_stmt(arm.body, root, writes, reads);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                self.collect_root_accesses_stmt(body, root, writes, reads)
            }
        }
    }

    fn declared_access_covers(
        declared_path: &[HirEffectSegment],
        used_fields: &[StrId],
        used_index: Option<&UsedIndex>,
    ) -> bool {
        let mut d_fields = Vec::new();
        let mut d_index: Option<&EffectIndexKey> = None;
        for seg in declared_path {
            match seg {
                HirEffectSegment::Field(f) => d_fields.push(*f),
                HirEffectSegment::Index(key) => d_index = Some(key),
            }
        }
        if d_fields != used_fields {
            return false;
        }
        match (d_index, used_index) {
            (None, _) => true,
            (Some(EffectIndexKey::Dynamic), _) => true, // can't correlate, so grant the whole slot space
            (Some(EffectIndexKey::Const(d_i)), Some(UsedIndex::Range(u_s, u_e))) => {
                *d_i <= *u_s && *u_e <= *d_i + 1
            }
            (
                Some(EffectIndexKey::Place { root, path }),
                Some(UsedIndex::Place(u_root, u_path)),
            ) => *root == *u_root && *path == u_path.as_slice(),
            _ => false,
        }
    }

    fn validate_multi_place_declaration(
        &mut self,
        root: StrId,
        declared: &[HirEffectAccess<'bump>],
        body: &HirStmt<'a, 'bump>,
    ) {
        let mut writes = Vec::new();
        let mut reads = Vec::new();
        self.collect_root_accesses_stmt(body, root, &mut writes, &mut reads);

        for (path, range) in &writes {
            let covered = declared.iter().any(|d| {
                d.ref_kind != RefKind::Shared
                    && Self::declared_access_covers(d.path, path, range.as_ref())
            });
            if !covered {
                self.record(TypeErrorKind::Generic(format!(
                    "writes to `{}{}` but the declared effects don't grant `&mut` access there",
                    self.str_id_to_string(root),
                    Self::path_display(path, range.as_ref()),
                )));
            }
        }
        for (path, range) in &reads {
            let covered = declared
                .iter()
                .any(|d| Self::declared_access_covers(d.path, path, range.as_ref()));
            if !covered {
                self.record(TypeErrorKind::Generic(format!(
                    "reads `{}{}` but it isn't listed in the declared effects",
                    self.str_id_to_string(root),
                    Self::path_display(path, range.as_ref()),
                )));
            }
        }
    }

    fn validate_multi_place_signature(
        &mut self,
        param_type: Option<&HirType<'a, 'bump>>,
        accesses: &[HirEffectAccess<'bump>],
    ) {
        let outer_mut = match param_type {
            Some(HirType::Ref { ref_kind, .. }) => *ref_kind != RefKind::Shared,
            Some(HirType::SafePointer {
                mutability_state, ..
            })
            | Some(HirType::UnsafePointer {
                mutability_state, ..
            }) => *mutability_state == MutabilityState::Mut,
            Some(HirType::This) => true,
            None => true,
            _ => {
                self.record(TypeErrorKind::Generic(
                    "`.{...}` access lists are only valid on reference or pointer parameters"
                        .to_string(),
                ));
                return;
            }
        };
        if !outer_mut && accesses.iter().any(|a| a.ref_kind == RefKind::Unique) {
            self.record(TypeErrorKind::Generic(
                "declares `&mut` access through a parameter that isn't itself `&mut`".to_string(),
            ));
        }
    }

    fn check_init_read_path(&mut self, root: StrId, path: &[StrId], root_str: &str) {
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

    fn status_at_path(node: &InitNode, path: &[StrId]) -> InitStatus {
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

    fn join_init_states(
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

    fn join_nodes(a: InitNode, b: InitNode) -> InitNode {
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

    pub fn scope_end_init_snapshot(&self, name: StrId) -> Option<&InitNode> {
        self.init_state.get(&name)
    }

    pub fn uninit_ty(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.uninit_backfill.get(&Self::expr_key(expr)).copied()
    }

    fn check_unsafe_call(&mut self, func: &HirFunc<'a, 'bump>, display_name: &str) {
        if matches!(func.function_metadata.func_safety, FuncSafety::Unsafe) && !self.in_unsafe() {
            self.record(TypeErrorKind::Generic(format!(
                "call to unsafe function `{}` requires an `unsafe` block",
                display_name
            )));
        }
    }

    fn analyze_definite_writes(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<bool> {
        if let Some(t) = self.write_templates.get(&func.name) {
            return t.clone();
        }

        self.write_templates.insert(func.name, Vec::new());
        let templates = Self::build_write_templates(func);
        self.write_templates.insert(func.name, templates.clone());
        templates
    }

    fn build_write_templates(func: &HirFunc<'a, 'bump>) -> Vec<bool> {
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

    fn analyze_write_stmt(
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

    fn analyze_write_block(
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

    fn expr_is_full_deref_write(expr: &HirExpr<'a, 'bump>, root: StrId) -> bool {
        match expr {
            HirExpr::Assignment { target, op, .. } => {
                matches!(op, AssignmentOperator::Assign) && Self::is_deref_of(target, root)
            }
            _ => false,
        }
    }

    fn is_deref_of(expr: &HirExpr<'a, 'bump>, root: StrId) -> bool {
        match expr {
            HirExpr::Deref { expr: inner, .. } => Self::is_ident(inner, root),
            _ => false,
        }
    }

    fn is_ident(expr: &HirExpr<'a, 'bump>, root: StrId) -> bool {
        match expr {
            HirExpr::Ident(name, _) => *name == root,
            _ => false,
        }
    }

    fn analyze_read_templates(&mut self, func: &HirFunc<'a, 'bump>) -> Vec<ReadTemplate> {
        if let Some(t) = self.read_templates.get(&func.name) {
            return t.clone();
        }
        let templates = Self::build_read_templates(func);
        self.read_templates.insert(func.name, templates.clone());
        templates
    }

    fn build_read_templates(func: &HirFunc<'a, 'bump>) -> Vec<ReadTemplate> {
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
            Self::collect_param_reads_stmt(&body, &param_index, has_this, &mut templates);
        }
        templates
    }

    fn record_param_read(
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

    fn collect_param_reads_write_target(
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
        Self::collect_param_reads_expr(target, param_index, has_this, templates);
    }

    fn mut_raw_ptr_cast_operand<'e>(
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

    fn collect_param_reads_expr(
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
                Self::collect_param_reads_expr(expr, param_index, has_this, templates);
                for arm in arms.iter() {
                    if let Some(guard) = arm.guard {
                        Self::collect_param_reads_expr(guard, param_index, has_this, templates);
                    }
                    Self::collect_param_reads_stmt(arm.body, param_index, has_this, templates);
                }
            }
            HirExpr::Block { body, .. } => {
                for s in body.iter() {
                    Self::collect_param_reads_stmt(s, param_index, has_this, templates);
                }
            }
            HirExpr::Range { start, end, .. } => {
                Self::collect_param_reads_expr(start, param_index, has_this, templates);
                Self::collect_param_reads_expr(end, param_index, has_this, templates);
            }
            HirExpr::Slice {
                object, start, end, ..
            } => {
                Self::collect_param_reads_expr(object, param_index, has_this, templates);
                Self::collect_param_reads_expr(start, param_index, has_this, templates);
                Self::collect_param_reads_expr(end, param_index, has_this, templates);
            }
            HirExpr::Tuple(exprs, _)
            | HirExpr::ArrayLiteral {
                elements: exprs, ..
            } => {
                for e in exprs.iter() {
                    Self::collect_param_reads_expr(e, param_index, has_this, templates);
                }
            }
            HirExpr::Binary { left, right, .. } | HirExpr::Comparison { left, right, .. } => {
                Self::collect_param_reads_expr(left, param_index, has_this, templates);
                Self::collect_param_reads_expr(right, param_index, has_this, templates);
            }
            HirExpr::Call { callee, args, .. } | HirExpr::InterfaceCall { callee, args, .. } => {
                Self::collect_param_reads_expr(callee, param_index, has_this, templates);
                for a in args.iter() {
                    if let Some(inner) = Self::mut_raw_ptr_cast_operand(a) {
                        // `param as [*]mut T` handed straight to a callee: not a read
                        // of the contents. Prefix loads are still recorded.
                        Self::collect_param_reads_write_target(
                            inner,
                            param_index,
                            has_this,
                            templates,
                        );
                    } else {
                        Self::collect_param_reads_expr(a, param_index, has_this, templates);
                    }
                }
            }
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                Self::collect_param_reads_expr(object, param_index, has_this, templates);
            }
            HirExpr::Assignment {
                target, op, value, ..
            } => {
                if matches!(op, AssignmentOperator::Assign) {
                    Self::collect_param_reads_write_target(
                        target,
                        param_index,
                        has_this,
                        templates,
                    );
                } else {
                    // compound assignment reads the old value
                    Self::collect_param_reads_expr(target, param_index, has_this, templates);
                }
                Self::collect_param_reads_expr(value, param_index, has_this, templates);
            }
            HirExpr::StructInit { args, .. } => {
                for f in args.iter() {
                    Self::collect_param_reads_expr(&f.value, param_index, has_this, templates);
                }
            }
            HirExpr::EnumInit { args, .. } => {
                for a in args.iter() {
                    Self::collect_param_reads_expr(a, param_index, has_this, templates);
                }
            }
            HirExpr::ExprList { list, .. } => {
                for e in list.iter() {
                    Self::collect_param_reads_expr(e, param_index, has_this, templates);
                }
            }
            HirExpr::Deref { expr, .. }
            | HirExpr::Cast { expr, .. }
            | HirExpr::Ref { expr, .. } => {
                Self::collect_param_reads_expr(expr, param_index, has_this, templates);
            }
            HirExpr::Index { object, index, .. } => {
                Self::collect_param_reads_expr(object, param_index, has_this, templates);
                Self::collect_param_reads_expr(index, param_index, has_this, templates);
            }
            HirExpr::InterpolatedString(parts) => {
                for p in parts.iter() {
                    if let InterpolationPart::Expr(e) = p {
                        Self::collect_param_reads_expr(e, param_index, has_this, templates);
                    }
                }
            }
            HirExpr::If { if_stmt, .. } => {
                Self::collect_param_reads_stmt(if_stmt, param_index, has_this, templates);
            }
            HirExpr::Intrinsic { args, .. } => {
                for a in args.iter() {
                    Self::collect_param_reads_expr(a, param_index, has_this, templates);
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

    fn collect_param_reads_stmt(
        stmt: &HirStmt<'a, 'bump>,
        param_index: &FxHashMap<StrId, usize>,
        has_this: bool,
        templates: &mut [ReadTemplate],
    ) {
        match stmt {
            HirStmt::Let {
                value,
                else_block,
                catch_pattern,
                ..
            } => {
                Self::collect_param_reads_expr(value, param_index, has_this, templates);
                if let Some(b) = else_block {
                    Self::collect_param_reads_stmt(b, param_index, has_this, templates);
                }
                if let Some(pattern) = catch_pattern {
                    match pattern {
                        HirErrorHandlerPattern::Single { body, .. } => {
                            for s in body.iter() {
                                Self::collect_param_reads_stmt(s, param_index, has_this, templates);
                            }
                        }
                        HirErrorHandlerPattern::Multiple { branches } => {
                            for branch in branches.iter() {
                                for s in branch.body.iter() {
                                    Self::collect_param_reads_stmt(
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
                Self::collect_param_reads_expr(&c.value, param_index, has_this, templates)
            }
            HirStmt::Return(Some(e), _) | HirStmt::Break(Some(e), _) => {
                Self::collect_param_reads_expr(e, param_index, has_this, templates)
            }
            HirStmt::Return(None, _)
            | HirStmt::Break(None, _)
            | HirStmt::Continue(_)
            | HirStmt::Import(..)
            | HirStmt::Package(..) => {}
            HirStmt::Expr(e) => Self::collect_param_reads_expr(e, param_index, has_this, templates),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span: _,
            } => {
                Self::collect_param_reads_expr(cond, param_index, has_this, templates);
                for s in then_block.iter() {
                    Self::collect_param_reads_stmt(s, param_index, has_this, templates);
                }
                if let Some(e) = else_block {
                    Self::collect_param_reads_stmt(e, param_index, has_this, templates);
                }
            }
            HirStmt::While { cond, body } => {
                Self::collect_param_reads_expr(cond, param_index, has_this, templates);
                Self::collect_param_reads_stmt(body, param_index, has_this, templates);
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(i) = init {
                    Self::collect_param_reads_stmt(i, param_index, has_this, templates);
                }
                if let Some(c) = condition {
                    Self::collect_param_reads_expr(c, param_index, has_this, templates);
                }
                if let Some(inc) = increment {
                    Self::collect_param_reads_expr(inc, param_index, has_this, templates);
                }
                Self::collect_param_reads_stmt(body, param_index, has_this, templates);
            }
            HirStmt::Block { body, span: _ } => {
                for s in body.iter() {
                    Self::collect_param_reads_stmt(s, param_index, has_this, templates);
                }
            }
            HirStmt::Match {
                expr,
                arms,
                span: _,
            } => {
                Self::collect_param_reads_expr(expr, param_index, has_this, templates);
                for arm in arms.iter() {
                    if let Some(g) = arm.guard {
                        Self::collect_param_reads_expr(g, param_index, has_this, templates);
                    }
                    Self::collect_param_reads_stmt(arm.body, param_index, has_this, templates);
                }
            }
            HirStmt::UnsafeBlock { body } | HirStmt::Defer(body) => {
                Self::collect_param_reads_stmt(body, param_index, has_this, templates)
            }
        }
    }

    fn resolve_template_place_from(
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

    fn read_template_touches_contents(template: &ReadTemplate) -> bool {
        match template {
            ReadTemplate::Opaque => true,
            ReadTemplate::Paths(paths) => paths.iter().any(|projections| {
                projections
                    .iter()
                    .any(|p| matches!(p, TemplateProjection::Index(_) | TemplateProjection::Deref))
            }),
        }
    }

    fn check_call_arg_read_effects(
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

    fn in_unsafe(&self) -> bool {
        self.unsafe_depth != 0
    }

    fn check_borrow_use_shell(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        place: PlaceId,
        kind: BorrowKind,
    ) {
        if let Err(e) = self.borrow_checker.check_use_shell(place, kind) {
            let provenance = self.infer_provenance(expr);
            let msg = self.describe_borrow_error(&e, provenance.as_ref());
            self.record(TypeErrorKind::Generic(msg));
        }
    }

    fn node_at_path_mut<'n>(node: &'n mut InitNode, path: &[StrId]) -> &'n mut InitNode {
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

    fn record_item_occurrence(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        ty: HirType<'a, 'bump>,
        declaring_module_idx: usize,
    ) {
        let Some((m, item_idx, tag)) = self
            .context
            .dep_graph
            .borrow()
            .resolve_item_in_module(declaring_module_idx, name)
        else {
            return;
        };
        self.occurrences.push((
            span,
            name,
            ty,
            self.context.current_module_idx,
            SymbolId::Item {
                module_idx: m,
                item_idx,
                tag,
            },
            false,
        ));
    }

    fn record_method_occurrence(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        ty: HirType<'a, 'bump>,
        target_type: StrId,
    ) {
        let Some((module_idx, item_idx, method_idx)) = self
            .context
            .dep_graph
            .borrow()
            .resolve_method(target_type, name)
        else {
            return;
        };
        self.occurrences.push((
            span,
            name,
            ty,
            self.context.current_module_idx,
            SymbolId::Method {
                module_idx,
                item_idx,
                method_idx,
            },
            false,
        ));
    }

    fn expr_references_local(&self, expr: &HirExpr<'a, 'bump>, local: StrId) -> bool {
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
            HirExpr::If { if_stmt, span: _ } => self.stmt_references_local(*if_stmt, local)
        }
    }

    fn stmt_references_local(&self, stmt: &HirStmt<'a, 'bump>, local: StrId) -> bool {
        match stmt {
            HirStmt::Let {
                value, else_block, ..
            } => {
                self.expr_references_local(value, local)
                    || else_block.is_some_and(|b| self.stmt_references_local(b, local))
            }
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

    fn mint_symbol_id(&mut self) -> SymbolId {
        let id = LocalSymbolId(self.next_symbol_id);
        self.next_symbol_id += 1;
        SymbolId::Local(id)
    }

    fn expr_key(expr: &HirExpr<'a, 'bump>) -> usize {
        expr as *const HirExpr<'a, 'bump> as usize
    }

    fn record_instance_args(&mut self, expr: &HirExpr<'a, 'bump>, args: &[HirType<'a, 'bump>]) {
        self.generic_instance_args
            .insert(Self::expr_key(expr), args.to_vec());
    }

    pub fn undefined_ty(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.undefined_backfill.get(&Self::expr_key(expr)).copied()
    }

    pub fn occurrences(
        &self,
    ) -> &[(
        SourceSpan<'a>,
        StrId,
        HirType<'a, 'bump>,
        usize,
        SymbolId,
        bool,
    )] {
        &self.occurrences
    }

    pub fn context(&self) -> &TypeContext<'a, 'bump> {
        &self.context
    }

    pub fn errors(&self) -> &[TypeError<'a>] {
        &self.errors
    }

    pub fn has_errors(&self) -> bool {
        !self.errors.is_empty()
    }

    pub fn take_errors(&mut self) -> Vec<TypeError<'a>> {
        std::mem::take(&mut self.errors)
    }

    fn set_span(&mut self, span: SourceSpan<'a>) {
        self.current_span = span;
    }

    fn slice_field_owned(ty: &HirType<'a, 'bump>) -> Option<bool> {
        let inner = match ty {
            HirType::Ref { inner, .. } | HirType::SafePointer { inner, .. } => *inner,
            other => other,
        };
        match inner {
            HirType::Slice(_) => Some(false),
            HirType::OwnedPointer { inner, .. } => match *inner {
                HirType::Slice(_) => Some(true),
                _ => None,
            },
            _ => None,
        }
    }

    fn record(&mut self, kind: TypeErrorKind) {
        if self.suppress_errors {
            return;
        }
        self.errors.push(kind.at(self.current_span));
    }

    fn recover<T>(&mut self, result: TypeCheckResult<'a, T>, fallback: T) -> T {
        match result {
            Ok(v) => v,
            Err(e) => {
                if !self.suppress_errors {
                    self.errors.push(e);
                }
                fallback
            }
        }
    }

    fn with_suppressed_errors<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let prev = self.suppress_errors;
        self.suppress_errors = true;
        f(self);
        self.suppress_errors = prev;
    }

    fn status_for_whole_struct(&self, node: &InitNode, struct_name: StrId) -> InitStatus {
        let InitNode::Struct(map) = node else {
            return match node {
                InitNode::Whole(s) => s.clone(),
                _ => InitStatus::Initialized,
            };
        };
        let struct_name_str = self.str_id_to_string(struct_name);
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

    fn node_at_path_ref<'n>(node: &'n InitNode, path: &[StrId]) -> &'n InitNode {
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

    fn converge_loop_states(
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

    fn is_zeroable(&self, ty: &HirType<'a, 'bump>) -> bool {
        match ty {
            HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::F32
            | HirType::F64 => true,

            HirType::Array(inner, _) => self.is_zeroable(inner),

            HirType::Tuple(elems) => elems.iter().all(|e| self.is_zeroable(e)),

            HirType::Struct { name, .. } => {
                let name_str = self.str_id_to_string(*name);
                match self.context.get_struct(&name_str) {
                    Some(def) => def.fields.iter().all(|f| self.is_zeroable(&f.field_type)),
                    None => false,
                }
            }

            HirType::Nullable(inner) => match **inner {
                // Pointer-shaped: all-zero bits legitimately means "null". Safe to zero-init.
                HirType::SafePointer { .. }
                | HirType::UnsafePointer { .. }
                | HirType::OwnedPointer { .. }
                | HirType::Ref { .. } => true,
                // Non-pointer nullable (e.g. i32?) needs a discriminant/tag, not just zero
                // bits
                // We could probably add some optimizations like `NonZero<u32>` like in Rust, but for now, this is good enough.
                _ => false,
            },

            // Impermissible: bool, char, string, enums, interfaces/dyn, lambdas,
            // pointers/refs, nullable, slices, generics, void/null/this/unknown.
            // Zeroing these either produces an invalid bit pattern (bool/char/enum
            // discriminants), a dangling/null reference where one shouldn't
            // silently appear (pointers), or is simply meaningless (lambda, dyn).
            _ => false,
        }
    }

    fn check_ident_init_read(&mut self, name: StrId, var_name: &str, ty: &HirType<'a, 'bump>) {
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

    pub fn register_module(&mut self, module: &HirModule<'a, 'bump>, module_idx: usize) {
        let prev_module_idx = self.context.current_module_idx;
        self.context.current_module_idx = module_idx;

        self.functions_by_module
            .entry(module_idx)
            .or_default()
            .clear();
        self.structs_by_module
            .entry(module_idx)
            .or_default()
            .clear();
        self.enums_by_module.entry(module_idx).or_default().clear();

        for item in module.items {
            match item {
                Hir::Struct(s) => {
                    let name = s.name.to_string();
                    self.context.add_struct(module_idx, name.clone(), **s);
                    self.structs_by_module
                        .entry(module_idx)
                        .or_default()
                        .insert(s.name);
                }
                Hir::Impl(i) => {
                    let target = i.target.to_string();
                    if let Some(methods) = i.methods {
                        let target_as_str = target.to_string();
                        let table = self.context.type_methods.entry(target_as_str).or_default();
                        for func in methods {
                            if table
                                .methods
                                .iter()
                                .any(|(name, _)| name == func.name.as_str())
                            {
                                self.current_span = func.span;
                                if self.suppress_errors {
                                    return;
                                }
                                self.errors.push(TypeErrorKind::Generic(format!(
                                    "function `{}` is already declared in this module with the same signature",
                                    func.unmangled_name
                                )).at(self.current_span));
                            }
                            table.insert(func.unmangled_name.to_string(), *func);
                        }
                    }
                    if let Some(interface) = i.interface {
                        self.context
                            .add_struct_interface(&target, interface.to_string());
                    }
                }
                Hir::Interface(i) => {
                    let name = i.name.to_string();
                    self.context.add_interface(module_idx, name, **i);
                }
                Hir::Enum(e) => {
                    let name = e.name.to_string();
                    self.context.add_enum(module_idx, name, **e);
                    self.enums_by_module
                        .entry(module_idx)
                        .or_default()
                        .insert(e.name);
                }
                Hir::Func(f) => {
                    let mangled_name = f.name.to_string();
                    let unmangled_name = f.unmangled_name.to_string();

                    if self
                        .functions_by_module
                        .get(&module_idx)
                        .is_some_and(|s| s.contains(&f.name))
                    {
                        self.set_span(f.span);
                        self.record(TypeErrorKind::Generic(format!(
                            "function `{}` is already declared in this module with the same signature",
                            unmangled_name
                        )));
                    }

                    self.context
                        .add_function(module_idx, unmangled_name.clone(), **f);
                    if mangled_name != unmangled_name {
                        self.context.add_function(module_idx, mangled_name, **f);
                    }
                    self.functions_by_module
                        .entry(module_idx)
                        .or_default()
                        .insert(f.name);
                }
                _ => {}
            }
        }

        let mut imports = ModuleImports {
            named: FxHashMap::default(),
            modules: std::collections::HashSet::new(),
            module_aliases: std::collections::HashMap::default(),
            wildcard: Vec::new(),
        };
        for import_path in module.imports {
            let Some(target_module) = self
                .context
                .dep_graph
                .borrow()
                .resolve_module_path(import_path.path)
            else {
                let path_str = import_path
                    .path
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                self.record(TypeErrorKind::Generic(format!(
                    "cannot resolve imported module `{}`",
                    path_str
                )));
                continue;
            };
            match import_path.member {
                Some(name) => {
                    imports.named.insert(name, target_module);
                }
                None => {
                    imports.modules.insert(target_module);
                    if let Some(&last_seg) = import_path.path.iter().last() {
                        imports.module_aliases.insert(last_seg, target_module);
                    }
                }
            }
        }

        let mut alias_targets: FxHashMap<StrId, usize> = FxHashMap::default();
        for auto_path in self.auto_imports.borrow().paths() {
            let segments: Vec<StrId> = auto_path
                .iter()
                .map(|s| StrId(self.context.string_pool.intern(s)))
                .collect();
            let Some(target_module) = self
                .context
                .dep_graph
                .borrow()
                .resolve_module_path(segments.as_slice())
            else {
                continue;
            };
            if target_module != module_idx && !imports.wildcard.contains(&target_module) {
                imports.wildcard.push(target_module);
            }

            if let Some(&last) = segments.last() {
                match alias_targets.get(&last) {
                    Some(&existing) if existing != target_module => {
                        let existing_pkg = self
                            .context
                            .dep_graph
                            .borrow()
                            .get_module_package(existing)
                            .map(|p| p.to_string())
                            .unwrap_or_default();
                        let new_pkg = self
                            .context
                            .dep_graph
                            .borrow()
                            .get_module_package(target_module)
                            .map(|p| p.to_string())
                            .unwrap_or_default();

                        // The fastest and most efficient way to avoid a borrow checker issue is to inline `self.record` :P
                        // Unlike Zeta, Rust cannot prove that this is safe and is too conservative.
                        // Don't use rust, use zeta!
                        if self.suppress_errors {
                            return;
                        }
                        self.errors.push(
                            TypeErrorKind::Generic(format!(
                                "auto-imported packages `{}` and `{}` both alias to `{}`; \
                             add an explicit `import` to disambiguate",
                                existing_pkg, new_pkg, last,
                            ))
                            .at(self.current_span),
                        );
                    }
                    _ => {
                        alias_targets.insert(last, target_module);
                    }
                }
            }
        }

        self.imports_by_module.insert(module_idx, imports);
        self.context.current_module_idx = prev_module_idx;
    }

    pub fn check_module_body(&mut self, module: &HirModule<'a, 'bump>, module_idx: usize) {
        self.occurrences
            .retain(|(_, _, _, m, _, _)| *m != module_idx);
        self.context.current_module_idx = module_idx;
        for item in module.items {
            if let Hir::Func(func) = item {
                self.check_function(func);
            }
            if let Hir::Struct(ty_struct) = item {
                let Some(struct_interfaces) = self
                    .context
                    .struct_interfaces
                    .get(&ty_struct.name.to_string())
                else {
                    continue;
                };

                if struct_interfaces.contains("Copy") && struct_interfaces.contains("Drop") {
                    self.record(TypeErrorKind::Generic(format!(
                        "{} should not implement Copy and Drop at the same time",
                        ty_struct.name
                    )));
                }
            }
        }
    }

    fn check_name_import_visibility(&mut self, name: StrId, name_str: &str) {
        let current = self.context.current_module_idx;

        if self
            .functions_by_module
            .get(&current)
            .is_some_and(|s| s.contains(&name))
        {
            return;
        }

        if let Some(imports) = self.imports_by_module.get(&current) {
            if let Some(&target_module) = imports.named.get(&name) {
                if self
                    .functions_by_module
                    .get(&target_module)
                    .is_some_and(|s| s.contains(&name))
                {
                    return;
                }
            }

            let candidates: Vec<usize> = imports
                .wildcard
                .iter()
                .copied()
                .filter(|m| {
                    self.functions_by_module
                        .get(m)
                        .is_some_and(|s| s.contains(&name))
                })
                .collect();

            if candidates.len() == 1 {
                return;
            }
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
                return;
            }
        }

        self.record(TypeErrorKind::Generic(format!(
            "`{}` is not declared in this module and has not been imported",
            name_str,
        )));
    }

    fn check_module_path_imported(&mut self, path_segments: &[StrId]) {
        let current = self.context.current_module_idx;
        let Some(target) = self
            .context
            .dep_graph
            .borrow()
            .resolve_module_path(path_segments)
        else {
            return;
        };
        if target == current {
            return;
        }

        let imported = self
            .imports_by_module
            .get(&current)
            .is_some_and(|imp| imp.modules.contains(&target));
        if !imported {
            let path_str = path_segments
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join("::");
            self.record(TypeErrorKind::Generic(format!(
                "module `{}` used without an `import {};` declaration",
                path_str, path_str,
            )));
        }
    }

    fn check_function(&mut self, func: &HirFunc<'a, 'bump>) {
        let mut func_context = self.context.create_child_scope();

        self.borrow_checker = BorrowChecker::new();
        self.borrow_checker.begin_scope();

        if let Some(params) = func.params {
            for param in params.iter() {
                match param {
                    HirParam::Normal {
                        name, param_type, ..
                    } => {
                        let param_name = self.str_id_to_string(*name);
                        let symbol_id = self.mint_symbol_id();
                        func_context.add_variable(param_name, *param_type, symbol_id);
                        self.borrow_checker.declare_local(*name);
                        self.mark_whole_init(*name);
                    }
                    HirParam::This { kind, .. } => {
                        let rk = match kind {
                            ThisPassingKind::RefConst
                            | ThisPassingKind::ConstSafePtr
                            | ThisPassingKind::ConstUnsafePtr => Some(RefKind::Shared),

                            ThisPassingKind::RefMut
                            | ThisPassingKind::MutSafePtr
                            | ThisPassingKind::MutUnsafePtr => Some(RefKind::Unique),

                            ThisPassingKind::MultiPlace => None,
                            ThisPassingKind::Move | ThisPassingKind::MoveMut => None, // owned

                            ThisPassingKind::RefAlias => Some(RefKind::Alias),
                        };
                        if let Some(rk) = rk {
                            self.local_ref_kind.insert(self.this_id, rk);
                        }

                        let self_ty = self.self_type_for_func(func);
                        let symbol_id = self.mint_symbol_id();
                        func_context.add_variable("this".to_string(), self_ty, symbol_id);
                        self.borrow_checker.declare_local(self.this_id);
                        self.mark_whole_init(self.this_id);
                    }
                }
            }

            for param in params.iter() {
                let (root, multi_place, param_type) = match param {
                    HirParam::This {
                        kind: ThisPassingKind::MultiPlace,
                        multi_place,
                        ..
                    } => (self.this_id, *multi_place, None),
                    HirParam::Normal {
                        name,
                        param_type,
                        multi_place,
                        ..
                    } => (*name, *multi_place, Some(param_type)),
                    _ => continue,
                };
                let Some(accesses) = multi_place else {
                    continue;
                };
                self.validate_multi_place_signature(param_type, accesses);
                if let Some(body) = func.body {
                    self.validate_multi_place_declaration(root, accesses, &body);
                }
            }
        }

        func_context.current_return_type = func.return_type;

        if let Some(body) = func.body {
            let (cfg, points) = CfgBuilder::new().build(&body);
            self.cfg = cfg;
            self.stmt_points = points.stmt_points;
            self.stmt_after_points = points.stmt_after_points;
            self.point_locals_used = FxHashMap::default();

            self.current_point = self.cfg.entry.unwrap_or_default();
            self.collect_locals_used_stmt(&body);

            self.current_point = self.cfg.entry.unwrap_or_default();

            let old_context = std::mem::replace(&mut self.context, func_context);
            self.check_stmt(&body);
            self.context = old_context;
        }

        self.check_return_provenance(func);

        self.borrow_checker.end_scope();
    }

    fn self_type_for_func(&self, func: &HirFunc<'a, 'bump>) -> HirType<'a, 'bump> {
        let Some(target) = func.impl_target else {
            return HirType::This;
        };
        let target_str = self.str_id_to_string(target);

        if let Some(def) = self.context.get_struct(&target_str) {
            let type_args: Vec<HirType<'a, 'bump>> = def
                .generics
                .unwrap_or(&[])
                .iter()
                .map(|g| HirType::Generic(g.name))
                .collect();
            let field_types: Vec<HirType<'a, 'bump>> =
                def.fields.iter().map(|f| f.field_type).collect();
            return HirType::Struct {
                name: target,
                field_types: self.context.bump.alloc_slice(&field_types),
                type_args: self.context.bump.alloc_slice_copy(&type_args),
            };
        }

        if self.context.get_interface(&target_str).is_some() {
            return HirType::DynInterface(target, &[]);
        }

        HirType::This
    }

    fn register_multi_place_loans(
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

    fn effect_index_key_to_bound(&mut self, key: &EffectIndexKey) -> Bound {
        match key {
            EffectIndexKey::Const(i) => Bound::Const(*i),
            EffectIndexKey::Place { root, path } => {
                let mut name = self.str_id_to_string(*root);
                for seg in path.iter() {
                    name.push('.');
                    name.push_str(&self.str_id_to_string(*seg));
                }
                Bound::Symbol(StrId(self.context.string_pool.intern(&name)))
            }
            EffectIndexKey::Dynamic => self.fresh_opaque(),
        }
    }

    fn collect_locals_used_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) {
        if let Some(&point) = self.stmt_points.get(&Self::stmt_key(stmt)) {
            self.current_point = point;
        }
        match stmt {
            HirStmt::Let {
                value,
                else_block,
                catch_pattern,
                ..
            } => {
                self.collect_locals_used_expr(value);
                if let Some(b) = else_block {
                    self.collect_locals_used_stmt(b);
                }
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

    fn collect_locals_used_expr(&mut self, expr: &HirExpr<'a, 'bump>) {
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
        }
    }

    fn stmt_key(stmt: &HirStmt<'a, 'bump>) -> usize {
        stmt as *const HirStmt<'a, 'bump> as usize
    }

    fn set_point(&mut self, stmt: &HirStmt<'a, 'bump>) {
        if let Some(&point) = self.stmt_points.get(&Self::stmt_key(stmt)) {
            self.current_point = point;
        }
    }

    fn check_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.set_point(stmt);
        match stmt {
            HirStmt::Let {
                name,
                ty,
                value,
                mutable,
                else_block,
                span,
                is_static: _,
                catch_pattern: _,
            } => {
                let var_name = self.str_id_to_string(*name);
                let is_wildcard = var_name == "_";

                if !is_wildcard && self.context.variables.contains_key(&var_name) {
                    self.record(TypeErrorKind::VariableAlreadyExists {
                        var_name: var_name.clone(),
                    });
                }

                let value_type = self.check_expr_expected(value, ty);

                let is_uninit_value = matches!(value, HirExpr::Uninit { .. });
                if !is_uninit_value {
                    self.check_and_record_value_use(value, &value_type);
                }

                if let Some(else_block) = else_block {
                    match &value_type {
                        HirType::Nullable(inner) => {
                            let inner = **inner;
                            let result = self.types_compatible(ty, &inner);
                            self.recover(result, ());

                            let else_context = self.context.create_child_scope();
                            let old_context = std::mem::replace(&mut self.context, else_context);
                            self.check_stmt(else_block);
                            self.context = old_context;
                        }
                        _ => {
                            self.record(TypeErrorKind::Generic(format!(
                                "`? else` used on non-nullable type `{}`",
                                self.type_to_string(&value_type)
                            )));
                        }
                    }
                } else {
                    let result = self.types_compatible(ty, &value_type);
                    self.recover(result, ());
                }

                if self.expr_is_dangling(value) {
                    self.context.mark_dangling(var_name.clone());
                }

                let symbol_id = self.mint_symbol_id();
                self.context
                    .add_variable_with_mutability(var_name, *ty, *mutable, symbol_id);
                self.borrow_checker.declare_local(*name);
                self.occurrences.push((
                    *span,
                    *name,
                    *ty,
                    self.context.current_module_idx,
                    symbol_id,
                    true,
                ));

                if is_uninit_value {
                    self.mark_whole_uninit(*name);
                } else {
                    self.mark_whole_init(*name);
                    if let HirExpr::StructInit { args, .. } = value {
                        for fi in args.iter() {
                            if matches!(fi.value, HirExpr::Uninit { .. }) {
                                self.mark_field_uninit(*name, &[fi.name]);
                            }
                        }
                    }
                }

                if matches!(
                    ty,
                    HirType::SafePointer { .. } | HirType::UnsafePointer { .. }
                ) {
                    if let Some(place) = self.resolve_place(value) {
                        if let Some(&(base, ref offset)) = self.borrow_checker.pointee_of(place) {
                            let declared = *self.borrow_checker.local_place(*name).unwrap();
                            self.borrow_checker
                                .record_pointee(declared, base, offset.clone());
                        }
                    }
                }

                if let Some(loan_id) = self.call_loans.remove(&Self::expr_key(value)) {
                    self.loan_owners.insert(loan_id, *name);
                    if let Some(loan) = self.borrow_checker.loan(loan_id) {
                        self.local_provenance_place.insert(*name, loan.place);
                    }
                } else if let HirExpr::Ref {
                    expr: ref_target, ..
                } = value
                {
                    if let Some(place) = self.resolve_place(ref_target) {
                        self.local_provenance_place.insert(*name, place);
                        if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                            self.loan_owners.insert(loan_id, *name);
                        }
                    }
                }

                None
            }
            HirStmt::Return(expr, span) => {
                self.set_span(*span);
                if let Some(e) = expr {
                    let expected_return = self.context.current_return_type;
                    let expr_type = match expected_return {
                        Some(ret) => self.check_expr_expected(e, &ret),
                        None => self.check_expr(e),
                    };
                    self.check_and_record_value_use(e, &expr_type);
                    let dangling = self.check_no_dangling_pointer(e);
                    self.recover(dangling, ());
                    if let Some(expected_return) = expected_return {
                        self.recover(self.types_compatible(&expected_return, &expr_type), ());
                    }
                } else if let Some(expected_return) = self.context.current_return_type {
                    if expected_return != HirType::Void {
                        self.record(TypeErrorKind::InvalidReturnType {
                            expected: self.type_to_string(&expected_return),
                            found: "void".to_string(),
                        });
                    }
                }
                Some(HirType::Never)
            }
            HirStmt::Expr(e) => {
                let snap = self.snapshot_call_loan_keys();
                let ty = self.check_expr(e);
                self.end_temp_call_loans(&snap);
                Some(ty)
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                self.set_span(*span);
                self.check_if_branches(cond, *then_block, *else_block, None)
            }
            HirStmt::While { cond, body } => {
                let cond_type = self.check_expr(cond);
                if cond_type != HirType::Boolean {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: "bool".to_string(),
                        found: self.type_to_string(&cond_type),
                    });
                }

                self.context.enter_loop();

                let entry_state = self.move_state.clone();
                let init_entry = self.init_state.clone();
                let (converged_entry, converged_init) =
                    self.converge_loop_states(body, entry_state, init_entry);
                self.move_state = converged_entry;
                self.init_state = converged_init;
                self.check_stmt(body);

                self.context.exit_loop();
                None
            }

            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                if let Some(init_stmt) = init {
                    self.check_stmt(init_stmt);
                }

                if let Some(cond) = condition {
                    let cond_type = self.check_expr(cond);
                    if cond_type != HirType::Boolean {
                        self.record(TypeErrorKind::TypeMismatch {
                            expected: "bool".to_string(),
                            found: self.type_to_string(&cond_type),
                        });
                    }
                }

                self.context.enter_loop();

                let entry_state = self.move_state.clone();
                let init_entry = self.init_state.clone();
                let (converged_entry, converged_init) =
                    self.converge_loop_states(body, entry_state, init_entry);
                self.move_state = converged_entry;
                self.init_state = converged_init;
                self.check_stmt(body);

                self.context.exit_loop();

                if let Some(inc) = increment {
                    self.check_expr(inc);
                }

                None
            }
            HirStmt::Block { body, span: _ } => {
                self.borrow_checker.begin_scope();
                let mut block_context = self.context.create_child_scope();

                let local_names: Vec<StrId> = body
                    .iter()
                    .filter_map(|s| match s {
                        HirStmt::Let { name, .. } => Some(*name),
                        _ => None,
                    })
                    .collect();

                let mut value = None;
                for (i, stmt) in body.iter().enumerate() {
                    let old_context = std::mem::replace(&mut self.context, block_context);
                    value = self.check_stmt(stmt);
                    block_context = self.context.clone();
                    self.context = old_context;

                    let after_point = self.stmt_after_points.get(&Self::stmt_key(stmt)).copied();

                    let dead_loans: Vec<LoanId> = self
                        .loan_owners
                        .iter()
                        .filter(|(_, owner)| local_names.contains(owner))
                        .filter(|(_, &owner)| match after_point {
                            Some(p) => !self.local_used_after(p, owner),
                            None => !body[(i + 1)..]
                                .iter()
                                .any(|s| self.stmt_references_local(s, owner)),
                        })
                        .map(|(&loan_id, _)| loan_id)
                        .collect();
                    for loan_id in dead_loans {
                        self.borrow_checker.end_loan_now(loan_id);
                        self.loan_owners.remove(&loan_id);
                    }
                }

                self.borrow_checker.end_scope();
                // sweep anything that reached scope-end without an early kill
                self.loan_owners
                    .retain(|_, owner| !local_names.contains(owner));
                for name in &local_names {
                    self.local_provenance_place.remove(name);
                    self.local_ref_kind.remove(name);
                }
                value
            }
            HirStmt::Break(expr, span) => {
                self.set_span(*span);
                if !self.context.in_loop {
                    self.record(TypeErrorKind::BreakOutsideLoop);
                }
                if let Some(e) = expr {
                    let expr_type = self.check_expr(e);
                    self.check_and_record_value_use(e, &expr_type);
                    if let Some(expected_return) = self.context.current_return_type {
                        let result = self.types_compatible(&expected_return, &expr_type);
                        self.recover(result, ());
                    }
                }
                Some(HirType::Never)
            }
            HirStmt::Continue(span) => {
                self.set_span(*span);
                if !self.context.in_loop {
                    self.record(TypeErrorKind::ContinueOutsideLoop);
                }
                Some(HirType::Never)
            }
            HirStmt::Const(const_stmt) => {
                let value_type = self.check_expr(&const_stmt.value);
                let result = self.types_compatible(&const_stmt.ty, &value_type);
                self.recover(result, ());
                let var_name = self.str_id_to_string(const_stmt.name);
                let symbol_id = self.mint_symbol_id();
                self.context
                    .add_variable(var_name, const_stmt.ty, symbol_id);
                None
            }
            HirStmt::Match { expr, arms, span } => {
                self.set_span(*span);
                Some(self.check_match_arms(expr, arms, None))
            }
            HirStmt::UnsafeBlock { body } => {
                self.unsafe_depth += 1;
                let gotten = self.check_stmt(body);
                self.unsafe_depth -= 1;
                gotten
            }
            HirStmt::Defer(hir_stmt) => self.check_stmt(hir_stmt),
            HirStmt::Import(path, span) => {
                self.set_span(*span);
                if self
                    .context
                    .dep_graph
                    .borrow()
                    .resolve_module_path(&path.path)
                    .is_none()
                {
                    let path_str = path
                        .path
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join("::");
                    self.record(TypeErrorKind::Generic(format!(
                        "cannot resolve imported module `{}`",
                        path_str
                    )));
                }
                None
            }
            HirStmt::Package(path, span) => {
                self.set_span(*span);
                if self
                    .context
                    .dep_graph
                    .borrow()
                    .resolve_module_path(&path.path)
                    .is_none()
                {
                    let path_str = path
                        .path
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join("::");
                    self.record(TypeErrorKind::Generic(format!(
                        "cannot resolve package path `{}`",
                        path_str
                    )));
                }
                None
            }
        }
    }

    fn condition_to_fact(&mut self, cond: &HirExpr<'a, 'bump>) -> Option<(Bound, Bound, bool)> {
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

    fn leaf_span(expr: &HirExpr<'a, 'bump>) -> Option<SourceSpan<'a>> {
        match expr {
            HirExpr::Number(_, s)
            | HirExpr::Null(s)
            | HirExpr::Decimal(_, s)
            | HirExpr::Boolean(_, s)
            | HirExpr::String(_, s)
            | HirExpr::Char(_, s)
            | HirExpr::Ident(_, s) => Some(*s),
            _ => None,
        }
    }

    fn check_expr_expected(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        expected: &HirType<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        match expr {
            HirExpr::Number(_, span) if self.is_integer(expected) => {
                self.set_span(*span);
                *expected
            }
            HirExpr::Decimal(_, span) if matches!(expected, HirType::F32 | HirType::F64) => {
                self.set_span(*span);
                *expected
            }
            HirExpr::Undefined {
                span,
                ty: HirType::Unknown,
            } => {
                self.set_span(*span);
                if !self.is_zeroable(expected) {
                    self.record(TypeErrorKind::Generic(format!(
                    "`undefined` cannot be used for type `{}`: it cannot be safely zero-initialized",
                    self.type_to_string(expected)
                )));
                    return HirType::Unknown;
                }
                self.undefined_backfill
                    .insert(Self::expr_key(expr), *expected);
                *expected
            }
            HirExpr::Uninit {
                span,
                ty: HirType::Unknown,
            } => {
                self.set_span(*span);
                self.uninit_backfill.insert(Self::expr_key(expr), *expected);
                *expected
            }
            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
            } => self.check_enum_init(
                expr,
                enum_name,
                variant,
                args,
                type_args,
                *span,
                Some(expected),
            ),

            HirExpr::Match {
                expr: scrutinee,
                arms,
                span,
            } => {
                self.set_span(*span);
                self.check_match_arms(scrutinee, arms, Some(expected))
            }

            HirExpr::If { if_stmt, span } => {
                self.set_span(*span);
                let HirStmt::If {
                    cond,
                    then_block,
                    else_block,
                    ..
                } = *if_stmt
                else {
                    unreachable!()
                };
                self.check_if_branches(&cond, then_block, *else_block, Some(expected))
                    .unwrap_or(HirType::Void)
            }

            HirExpr::Block {
                body,
                is_unsafe,
                span,
            } => {
                self.set_span(*span);
                if *is_unsafe {
                    self.unsafe_depth += 1;
                }
                let ret_ty = self
                    .check_block_body(body, Some(expected))
                    .unwrap_or(HirType::Void);
                if *is_unsafe {
                    self.unsafe_depth -= 1;
                }
                ret_ty
            }

            _ => self.check_expr(expr),
        }
    }

    fn check_expr_as_place(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        let prev = self.in_place_context;
        self.in_place_context = true;
        let ty = self.check_expr_as_place_inner(expr);
        self.in_place_context = prev;
        ty
    }

    fn check_expr_as_place_inner(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        match expr {
            HirExpr::Ident(name, span) => {
                let var_name = self.str_id_to_string(*name);
                let (symbol_id, ty) = match self.context.get_variable(&var_name) {
                    Some(ty) => ty,
                    None => {
                        self.record(TypeErrorKind::UndefinedVariable(var_name));
                        (SymbolId::Local(LocalSymbolId(u32::MAX)), HirType::Unknown)
                    }
                };
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
            HirExpr::FieldAccess {
                object,
                field,
                span,
            }
            | HirExpr::Get {
                object,
                field,
                span,
            } => {
                self.set_span(*span);
                self.check_field_access_no_init_check(object, *field)
            }
            HirExpr::Index {
                object,
                index,
                span,
            } => {
                self.set_span(*span);
                let object_ty = self.check_expr_suppressed(object);
                let index_ty = self.check_expr(index);
                self.recover(self.types_compatible(&HirType::I64, &index_ty), ());
                match object_ty {
                    HirType::SafePointer { inner, .. } | HirType::UnsafePointer { inner, .. } => {
                        if !self.in_unsafe() {
                            self.record(TypeErrorKind::Generic(
                                "indexing a raw/unsafe pointer requires an unsafe block"
                                    .to_string(),
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
                                self.type_to_string(&object_ty)
                            )));
                            HirType::Unknown
                        }
                    },
                }
            }
            HirExpr::Slice {
                object,
                start,
                end,
                inclusive: _,
                span,
            } => {
                self.set_span(*span);
                // Consumed here so it can't leak into nested expressions.
                let skip_init = std::mem::take(&mut self.skip_slice_init_check);
                let object_ty = self.check_expr_suppressed(object);
                let start_ty = self.check_expr(start);
                let end_ty = self.check_expr(end);
                if !self.is_integer(&start_ty) || !self.is_integer(&end_ty) {
                    self.record(TypeErrorKind::Generic(
                        "slice bounds must be integers".to_string(),
                    ));
                }
                if !skip_init && matches!(*Self::strip_ref(&object_ty), HirType::Array(..)) {
                    self.check_slice_range_init(object, start, end);
                }
                match *Self::strip_ref(&object_ty) {
                    HirType::Array(inner, _) | HirType::Slice(inner) => HirType::Slice(inner),
                    _ => {
                        self.record(TypeErrorKind::Generic(format!(
                            "cannot slice type `{}`",
                            self.type_to_string(&object_ty)
                        )));
                        HirType::Unknown
                    }
                }
            }
            _ => self.check_expr(expr),
        }
    }

    fn check_field_access_no_init_check(
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
            if self.str_id_to_string(field) == "len" {
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
                self.type_to_string(&obj_type)
            )));
            return HirType::Unknown;
        };

        let struct_name_str = self.str_id_to_string(struct_name);
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

        let field_name = self.str_id_to_string(field);
        let Some(field_idx) = struct_def
            .fields
            .iter()
            .position(|f| self.str_id_to_string(f.name) == field_name)
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

    fn check_expr(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        if let Some(span) = Self::leaf_span(expr) {
            self.set_span(span);
        }
        match expr {
            HirExpr::Number(_, _) => HirType::I64,
            HirExpr::Null(_) => HirType::Null,
            HirExpr::Decimal(_, _) => HirType::F64,
            HirExpr::Boolean(_, _) => HirType::Boolean,
            HirExpr::String(_, _) => HirType::String,
            HirExpr::Uninit { span, ty } => {
                self.set_span(*span);
                match ty {
                    HirType::Unknown => {
                        self.record(TypeErrorKind::TypeCannotBeInferred);
                        HirType::Unknown
                    }
                    other_type => *other_type,
                }
            }
            HirExpr::Undefined { span, ty } => {
                self.set_span(*span);
                match ty {
                    HirType::Unknown => {
                        self.record(TypeErrorKind::TypeCannotBeInferred);
                        HirType::Unknown
                    }
                    other_type => {
                        if !self.is_zeroable(other_type) {
                            self.record(TypeErrorKind::Generic(format!(
                                "`undefined` cannot be used for type `{}`: it cannot be safely zero-initialized",
                                self.type_to_string(other_type)
                            )));
                            return HirType::Unknown;
                        }
                        *other_type
                    }
                }
            }

            HirExpr::Ident(name, span) => {
                let var_name = self.str_id_to_string(*name);
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
            HirExpr::Tuple(exprs, _span) => {
                let mut types = Vec::new();
                for e in *exprs {
                    types.push(self.check_expr(e));
                }
                HirType::Tuple(self.context.bump.alloc_slice_copy(types.as_slice()))
            }
            HirExpr::Binary {
                left,
                op,
                right,
                span,
            } => {
                self.set_span(*span);
                let left_type = self.check_expr(left);
                let right_type = self.check_expr(right);
                let result = self.check_binary_op(&left_type, op, &right_type);
                self.recover(result, HirType::Unknown)
            }
            HirExpr::Intrinsic {
                kind,
                type_args,
                args,
                span,
            } => {
                self.set_span(*span);

                match kind {
                    IntrinsicKind::Replace => {
                        if !type_args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "$replace takes no type arguments".to_string(),
                            ));
                        }
                        if args.len() != 2 {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args: 2,
                                found_args: args.len(),
                            });
                            return HirType::Unknown;
                        }

                        let place_expr: &HirExpr<'a, 'bump> = match &args[0] {
                            HirExpr::Ref { expr, .. } => expr,
                            other => other,
                        };
                        if !matches!(
                            place_expr,
                            HirExpr::Ident(..)
                                | HirExpr::FieldAccess { .. }
                                | HirExpr::Get { .. }
                                | HirExpr::Deref { .. }
                                | HirExpr::Index { .. }
                        ) {
                            self.record(TypeErrorKind::Generic(
                            "$replace's first argument must be a place (variable, field, deref, or index)"
                                .to_string(),
                        ));
                            return HirType::Unknown;
                        }
                        if matches!(&args[1], HirExpr::Uninit { .. }) {
                            self.record(TypeErrorKind::Generic(
                            "$replace cannot write `uninit`: the slot must always hold a valid value".to_string(),
                        ));
                            return HirType::Unknown;
                        }
                        if let HirExpr::FieldAccess { object, field, .. }
                        | HirExpr::Get { object, field, .. } = place_expr
                        {
                            let f = field.as_str();
                            if (f == "len" || f == "cap")
                                && Self::slice_field_owned(&self.peek_type(object)).is_some()
                            {
                                self.record(TypeErrorKind::Generic(
                                    "$replace cannot target a slice's `.len`/`.cap`".to_string(),
                                ));
                                return HirType::Unknown;
                            }
                        }

                        let target_ty = self.check_expr_as_place(place_expr);
                        // The old value is read out, so the slot must be initialized.
                        self.check_read_for_compound_target(place_expr, &target_ty);

                        let value_ty = self.check_expr_expected(&args[1], &target_ty);
                        self.check_and_record_value_use(&args[1], &value_ty); // moves the new value in
                        self.recover(self.types_compatible(&target_ty, &value_ty), ());

                        if let HirExpr::Ident(name, _) = place_expr {
                            let var_name = self.str_id_to_string(*name);
                            if self.context.is_local_binding(&var_name)
                                && !self.context.is_mutable(&var_name)
                            {
                                self.record(TypeErrorKind::Generic(format!(
                                    "cannot $replace `{}`: it is not declared `mut`",
                                    var_name
                                )));
                            }
                        }
                        if let Some(place) = self.resolve_place(place_expr) {
                            self.check_borrow_use(place_expr, place, BorrowKind::Mutable);
                        }

                        if let Some((root, path)) = self.static_field_path(place_expr) {
                            if matches!(value_ty, HirType::Nullable(_) | HirType::Null) {
                                self.clear_non_null(root, &path);
                            } else {
                                self.mark_non_null(root, &path);
                            }
                            self.mark_field_init(root, &path);
                        }

                        target_ty // the old value, moved out to the caller
                    }
                    IntrinsicKind::Reinterpret => {
                        if type_args.len() != 1 {
                            self.record(TypeErrorKind::Generic(format!(
                                "$reinterpret expects exactly 1 type argument, found {}",
                                type_args.len()
                            )));
                            return HirType::Unknown;
                        }
                        if args.len() != 1 {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args: 1,
                                found_args: args.len(),
                            });
                            return HirType::Unknown;
                        }
                        let target_ty = type_args[0];
                        let source_ty = self.check_expr(&args[0]);
                        self.check_and_record_value_use(&args[0], &source_ty);
                        if !self.in_unsafe() {
                            self.record(TypeErrorKind::Generic(
                            "$reinterpret requires an unsafe block: it reinterprets a value's bit \
                             representation as a different type with no conversion or validation"
                                .to_string(),
                        ));
                        }
                        target_ty
                    }
                    IntrinsicKind::Unreachable => {
                        if !type_args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "$unreachable takes no type arguments".to_string(),
                            ));
                        }
                        if !args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "$unreachable takes no value arguments".to_string(),
                            ));
                        }
                        HirType::Never
                    }
                    IntrinsicKind::SizeOf | IntrinsicKind::AlignOf | IntrinsicKind::TypeName => {
                        if type_args.len() != 1 {
                            self.record(TypeErrorKind::Generic(format!(
                                "intrinsic expects exactly 1 type argument, found {}",
                                type_args.len()
                            )));
                        }
                        if !args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "this intrinsic takes no value arguments".to_string(),
                            ));
                        }
                        match kind {
                            IntrinsicKind::SizeOf | IntrinsicKind::AlignOf => HirType::Usize,
                            IntrinsicKind::TypeName => HirType::String,
                            _ => unreachable!(),
                        }
                    }
                    IntrinsicKind::Own => {
                        if !type_args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "$own takes no type arguments".to_string(),
                            ));
                        }
                        if args.is_empty() || args.len() > 4 {
                            self.record(TypeErrorKind::Generic(format!(
                                "$own expects 1 argument (ptr), 2 (ptr, allocator or len), 3 (ptr, allocator, len), \
                                 or 4 (ptr, allocator, len, cap) for owned slices, found {}",
                                args.len()
                            )));
                            return HirType::Unknown;
                        }

                        let ptr_ty = self.check_expr(&args[0]);
                        self.check_and_record_value_use(&args[0], &ptr_ty);
                        let pointee = match Self::strip_ref(&ptr_ty) {
                            HirType::SafePointer { inner, .. }
                            | HirType::UnsafePointer { inner, .. } => *inner,
                            _ => {
                                self.record(TypeErrorKind::Generic(format!(
                                    "$own expects a `*T` or `[*]T`, found `{}`",
                                    self.type_to_string(&ptr_ty)
                                )));
                                return HirType::Unknown;
                            }
                        };

                        let (alloc_arg, len_arg, cap_arg) = if args.len() == 4 {
                            (Some(&args[1]), Some(&args[2]), Some(&args[3]))
                        } else if args.len() == 3 {
                            (Some(&args[1]), Some(&args[2]), None)
                        } else if args.len() == 2 {
                            let arg1_ty = self.check_expr(&args[1]);
                            if self.is_integer(&arg1_ty) {
                                (None, Some(&args[1]), None)
                            } else {
                                (Some(&args[1]), None, None)
                            }
                        } else {
                            (None, None, None)
                        };

                        let allocator = if let Some(alloc_expr) = alloc_arg {
                            let alloc_ty = self.check_expr(alloc_expr);
                            self.check_and_record_value_use(alloc_expr, &alloc_ty);
                            let Some(allocator) = self.infer_provenance(alloc_expr) else {
                                self.record(TypeErrorKind::Generic(
                                    "$own's allocator argument must be a place with stable provenance (a global, `this`, a param, or a projection through them)".to_string(),
                                ));
                                return HirType::Unknown;
                            };

                            let alloc_struct_name = match Self::strip_ref(&alloc_ty) {
                                HirType::Struct { name, .. } => self.str_id_to_string(*name),
                                _ => {
                                    self.record(TypeErrorKind::Generic(
                                        "$own's allocator argument must be a struct implementing RawAllocator".to_string(),
                                    ));
                                    return HirType::Unknown;
                                }
                            };
                            if !self
                                .context
                                .struct_implements(&alloc_struct_name, "RawAllocator")
                            {
                                self.record(TypeErrorKind::Generic(format!(
                                    "`{}` does not implement `RawAllocator`",
                                    alloc_struct_name
                                )));
                            }

                            if !matches!(allocator.root, ProvenanceRoot::Global { .. }) {
                                if let Some(place) = self.resolve_place(alloc_expr) {
                                    match self.borrow_checker.borrow_shared(place) {
                                        Ok(loan_id) => {
                                            self.call_loans.insert(Self::expr_key(expr), loan_id);
                                        }
                                        Err(e) => {
                                            let msg =
                                                self.describe_borrow_error(&e, Some(&allocator));
                                            self.record(TypeErrorKind::Generic(msg));
                                        }
                                    }
                                }
                            }
                            allocator
                        } else {
                            let this_expr = HirExpr::This {
                                span: self.current_span,
                            };
                            let Some(allocator) = self.infer_provenance(&this_expr) else {
                                self.record(TypeErrorKind::Generic(
                                    "$own without explicit allocator requires a `this` allocator in scope".to_string(),
                                ));
                                return HirType::Unknown;
                            };
                            allocator
                        };

                        let result_inner = if let Some(cap_expr) = cap_arg {
                            let len_expr = len_arg.expect(
                                "cap_arg implies len_arg due to the 4-arg-only branch above",
                            );
                            let len_ty = self.check_expr(len_expr);
                            self.check_and_record_value_use(len_expr, &len_ty);
                            if !self.is_integer(&len_ty) {
                                self.record(TypeErrorKind::Generic(format!(
                                    "$own's len argument must be an integer, found `{}`",
                                    self.type_to_string(&len_ty)
                                )));
                            }
                            let cap_ty = self.check_expr(cap_expr);
                            self.check_and_record_value_use(cap_expr, &cap_ty);
                            if !self.is_integer(&cap_ty) {
                                self.record(TypeErrorKind::Generic(format!(
                                    "$own's cap argument must be an integer, found `{}`",
                                    self.type_to_string(&cap_ty)
                                )));
                            }
                            HirType::Slice(self.context.bump.alloc_value(pointee))
                        } else if let Some(len_expr) = len_arg {
                            let len_ty = self.check_expr(len_expr);
                            self.check_and_record_value_use(len_expr, &len_ty);
                            if !self.is_integer(&len_ty) {
                                self.record(TypeErrorKind::Generic(format!(
                                    "$own's argument must be an integer, found `{}`",
                                    self.type_to_string(&len_ty)
                                )));
                            }
                            self.record(TypeErrorKind::Generic(
                                "$own for an owned slice requires both `len` and `cap`: use \
                                 `$own(ptr, allocator, len, cap)`"
                                    .to_string(),
                            ));
                            HirType::Slice(self.context.bump.alloc_value(pointee))
                        } else {
                            *pointee
                        };

                        HirType::OwnedPointer {
                            inner: self.context.bump.alloc_value(result_inner),
                            allocator: Some(allocator),
                        }
                    }
                    IntrinsicKind::AssertAlign => {
                        if !type_args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "$assert_align takes no type arguments".to_string(),
                            ));
                        }
                        if args.len() != 2 {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args: 2,
                                found_args: args.len(),
                            });
                        } else {
                            let ptr_ty = self.check_expr(&args[0]);
                            self.check_and_record_value_use(&args[0], &ptr_ty);
                            if !matches!(
                                Self::strip_ref(&ptr_ty),
                                HirType::SafePointer { .. }
                                    | HirType::UnsafePointer { .. }
                                    | HirType::OwnedPointer { .. }
                            ) {
                                self.record(TypeErrorKind::Generic(format!(
                                    "$assert_align expects a pointer, found `{}`",
                                    self.type_to_string(&ptr_ty)
                                )));
                            }

                            let align_ty = self.check_expr(&args[1]);
                            if !self.is_integer(&align_ty) {
                                self.record(TypeErrorKind::Generic(format!(
                                    "$assert_align expects an integer alignment, found `{}`",
                                    self.type_to_string(&align_ty)
                                )));
                            }
                            if let HirExpr::Number(n, _) = &args[1] {
                                if *n <= 0 || (*n as u64) & ((*n as u64) - 1) != 0 {
                                    self.record(TypeErrorKind::Generic(format!(
                                        "alignment must be a positive power of two, found {}",
                                        n
                                    )));
                                }
                            }
                        }
                        HirType::Void
                    }
                    IntrinsicKind::AtomicCasU32 => {
                        if !self.in_unsafe() {
                            self.record(TypeErrorKind::Generic(
                                "$atomic_cas_u32 requires an unsafe block".to_string(),
                            ));
                        }
                        if args.len() != 3 {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args: 3,
                                found_args: args.len(),
                            });
                            return HirType::U32;
                        }
                        let ptr_ty = self.check_expr(&args[0]);
                        self.check_and_record_value_use(&args[0], &ptr_ty);
                        let points_to_u32 = matches!(
                            Self::strip_ref(&ptr_ty),
                            HirType::SafePointer { inner, .. } | HirType::UnsafePointer { inner, .. }
                                if matches!(**inner, HirType::U32)
                        );
                        if !points_to_u32 {
                            self.record(TypeErrorKind::Generic(format!(
                                "$atomic_cas_u32 expects a pointer to `u32`, found `{}`",
                                self.type_to_string(&ptr_ty)
                            )));
                        }
                        for a in &args[1..] {
                            let t = self.check_expr(a);
                            self.check_and_record_value_use(a, &t);
                            if !matches!(t, HirType::U32) {
                                self.record(TypeErrorKind::TypeMismatch {
                                    expected: "u32".to_string(),
                                    found: self.type_to_string(&t),
                                });
                            }
                        }
                        HirType::U32
                    }

                    IntrinsicKind::AtomicLoadU32 => {
                        if !self.in_unsafe() {
                            self.record(TypeErrorKind::Generic(
                                "$atomic_load_u32 requires an unsafe block".to_string(),
                            ));
                        }
                        if args.len() != 1 {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args: 1,
                                found_args: args.len(),
                            });
                            return HirType::U32;
                        }
                        let ptr_ty = self.check_expr(&args[0]);
                        self.check_and_record_value_use(&args[0], &ptr_ty);
                        let points_to_u32 = matches!(
                            Self::strip_ref(&ptr_ty),
                            HirType::SafePointer { inner, .. } | HirType::UnsafePointer { inner, .. }
                                if matches!(**inner, HirType::U32)
                        );
                        if !points_to_u32 {
                            self.record(TypeErrorKind::Generic(format!(
                                "$atomic_load_u32 expects a pointer to `u32`, found `{}`",
                                self.type_to_string(&ptr_ty)
                            )));
                        }
                        HirType::U32
                    }

                    IntrinsicKind::AtomicStoreU32 => {
                        if !self.in_unsafe() {
                            self.record(TypeErrorKind::Generic(
                                "$atomic_store_u32 requires an unsafe block".to_string(),
                            ));
                        }
                        if args.len() != 2 {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args: 2,
                                found_args: args.len(),
                            });
                            return HirType::Void;
                        }
                        let ptr_ty = self.check_expr(&args[0]);
                        self.check_and_record_value_use(&args[0], &ptr_ty);
                        let points_to_u32 = matches!(
                            Self::strip_ref(&ptr_ty),
                            HirType::SafePointer { inner, .. } | HirType::UnsafePointer { inner, .. }
                                if matches!(**inner, HirType::U32)
                        );
                        if !points_to_u32 {
                            self.record(TypeErrorKind::Generic(format!(
                                "$atomic_store_u32 expects a pointer to `u32`, found `{}`",
                                self.type_to_string(&ptr_ty)
                            )));
                        }
                        let val_ty = self.check_expr(&args[1]);
                        self.check_and_record_value_use(&args[1], &val_ty);
                        if !matches!(val_ty, HirType::U32) {
                            self.record(TypeErrorKind::TypeMismatch {
                                expected: "u32".to_string(),
                                found: self.type_to_string(&val_ty),
                            });
                        }
                        HirType::Void
                    }

                    IntrinsicKind::CpuRelax => {
                        if !args.is_empty() {
                            self.record(TypeErrorKind::Generic(
                                "$cpu_relax takes no arguments".to_string(),
                            ));
                        }
                        HirType::Void
                    }
                }
            }
            HirExpr::If { if_stmt, span } => {
                self.set_span(*span);
                let HirStmt::If { else_block, .. } = if_stmt else {
                    unreachable!()
                };
                if else_block.is_none() {
                    self.record(TypeErrorKind::Generic(
                        "if used as an expression must have an else branch".to_string(),
                    ));
                }
                self.check_stmt(if_stmt).unwrap_or(HirType::Void)
            }

            HirExpr::Match { expr, arms, span } => {
                self.set_span(*span);
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
                                found: self.type_to_string(&guard_type),
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

            HirExpr::Block {
                body,
                is_unsafe,
                span,
            } => {
                self.set_span(*span);
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

            HirExpr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                self.set_span(*span);
                let start_ty = self.check_expr(start);
                let end_ty = self.check_expr(end);
                self.check_and_record_value_use(start, &start_ty);
                self.check_and_record_value_use(end, &end_ty);
                if !self.is_integer(&start_ty) {
                    self.record(TypeErrorKind::Generic(format!(
                        "range bounds must be integers, found `{}`",
                        self.type_to_string(&start_ty)
                    )));
                }
                let result = self.types_compatible(&start_ty, &end_ty);
                self.recover(result, ());
                HirType::Range {
                    elem: self.context.bump.alloc_value(start_ty),
                    inclusive: *inclusive,
                }
            }

            HirExpr::Slice {
                object,
                start,
                end,
                inclusive: _,
                span,
            } => {
                self.set_span(*span);
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
                            self.type_to_string(&object_ty)
                        )));
                        HirType::Unknown
                    }
                }
            }
            HirExpr::Call {
                callee,
                args,
                span,
                type_args,
            } => match &callee {
                HirExpr::Ident(func_name, ident_span) => {
                    self.set_span(*ident_span);
                    let lookup_name = self.str_id_to_string(*func_name);

                    if self.context.is_local_binding(&lookup_name) {
                        let callee_type = self.check_expr(callee);
                        return match callee_type {
                            HirType::Lambda { return_type, .. } => *return_type,
                            _ => {
                                self.record(TypeErrorKind::Generic(format!(
                                    "Expression of type `{}` is not callable",
                                    self.type_to_string(&callee_type)
                                )));
                                HirType::Unknown
                            }
                        };
                    }

                    let func = match self.context.get_function(&lookup_name) {
                        Some(f) => f,
                        None => {
                            self.record(TypeErrorKind::UndefinedFunction(lookup_name));
                            return HirType::Unknown;
                        }
                    };
                    self.check_unsafe_call(&func, &lookup_name);

                    self.record_item_occurrence(
                        *ident_span,
                        *func_name,
                        func.return_type.unwrap_or(HirType::Void),
                        func.declaring_module_idx,
                    );

                    self.recover(
                        self.check_visibility(
                            func.function_metadata.visibility,
                            func.declaring_module_idx,
                            "function",
                            &lookup_name,
                        ),
                        (),
                    );
                    self.check_name_import_visibility(*func_name, &lookup_name);

                    let substitutions: FxHashMap<StrId, HirType<'a, 'bump>> = match func.generics {
                        Some(tp) if !tp.is_empty() => match type_args {
                            Some(ta) => {
                                let mut full_args = ta.to_vec();
                                if full_args.len() < tp.len() {
                                    let mut temp_map = FxHashMap::default();
                                    for (&p, &a) in tp.iter().zip(full_args.iter()) {
                                        temp_map.insert(p.name, a);
                                    }
                                    let start = full_args.len();
                                    for param in &tp[start..] {
                                        if let Some(ref def_ty) = param.default_type {
                                            let resolved =
                                                self.substitute_type_local(def_ty, &temp_map);
                                            temp_map.insert(param.name, resolved);
                                            full_args.push(resolved);
                                        } else {
                                            break;
                                        }
                                    }
                                }
                                if full_args.len() != tp.len() {
                                    self.record(TypeErrorKind::Generic(format!(
                                        "function `{}` expects {} type argument(s), found {}",
                                        lookup_name,
                                        tp.len(),
                                        ta.len()
                                    )));
                                }
                                let mut map = FxHashMap::default();
                                tp.iter()
                                    .zip(full_args.iter())
                                    .map(|(&p, &a)| (p, a))
                                    .for_each(|(p, a)| {
                                        map.insert(p.name, a);
                                    });
                                map
                            }
                            None => {
                                let mut full_args = Vec::new();
                                let mut temp_map = FxHashMap::default();
                                for param in tp.iter() {
                                    if let Some(ref def_ty) = param.default_type {
                                        let resolved =
                                            self.substitute_type_local(def_ty, &temp_map);
                                        temp_map.insert(param.name, resolved);
                                        full_args.push(resolved);
                                    } else {
                                        break;
                                    }
                                }
                                if full_args.len() == tp.len() {
                                    temp_map
                                } else {
                                    self.record(TypeErrorKind::Generic(format!(
                                        "generic function `{}` requires explicit type arguments, e.g. `{}<Type>(...)`",
                                        lookup_name, lookup_name
                                    )));
                                    FxHashMap::default()
                                }
                            }
                        },
                        _ => {
                            if type_args.is_some() {
                                self.record(TypeErrorKind::Generic(format!(
                                    "function `{}` is not generic; no type arguments expected",
                                    lookup_name
                                )));
                            }
                            FxHashMap::default()
                        }
                    };

                    let expected_args = func.params.map(|p| p.len()).unwrap_or(0);
                    if args.len() != expected_args {
                        self.record(TypeErrorKind::InvalidFunctionCall {
                            expected_args,
                            found_args: args.len(),
                        });
                    }

                    let Some(params) = func.params else {
                        let ret_ty = func.return_type.unwrap_or(HirType::Void);
                        return if substitutions.is_empty() {
                            ret_ty
                        } else {
                            self.substitute_type_local(&ret_ty, &substitutions)
                        };
                    };

                    let params = if substitutions.is_empty() {
                        params
                    } else {
                        self.substitute_params_local(params, &substitutions)
                    };

                    let unsubstituted_ret_ty = func.return_type.unwrap_or(HirType::Void);
                    let ret_ty = if substitutions.is_empty() {
                        unsubstituted_ret_ty
                    } else {
                        self.substitute_type_local(&unsubstituted_ret_ty, &substitutions)
                    };

                    if let Some(value) =
                        self.check_potential_this_param_for_move(args, func, params, ret_ty)
                    {
                        return value;
                    }

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

                    let arg_loans = self.check_all_func_args(args, params, None, Some(func));

                    if !self.return_type_may_alias(&ret_ty) {
                        for loan in arg_loans {
                            self.borrow_checker.end_loan_now(loan);
                        }
                    }

                    ret_ty
                }

                HirExpr::FieldAccess {
                    object,
                    field,
                    span,
                } => {
                    self.set_span(*span);
                    let obj_type = self.check_expr(object);
                    let stripped = Self::strip_ref(&obj_type);

                    if let Some(elem) = match *stripped {
                        HirType::Slice(e) | HirType::Array(e, _) => Some(*e),
                        _ => None,
                    } {
                        let name = field.as_str();
                        if SLICE_PRIMITIVES.contains(&name) {
                            return self.check_slice_primitive_call(object, elem, name, args);
                        }
                    }

                    let interface_name = match stripped {
                        HirType::DynInterface(name, _) => Some(name.to_string()),
                        HirType::Dyn { bounds } => bounds.iter().find_map(|b| match b {
                            HirType::DynInterface(name, _) => Some(name.to_string()),
                            HirType::Struct { name, .. } => {
                                let name_str = name.to_string();
                                self.context.get_interface(&name_str).map(|_| name_str)
                            }
                            _ => None,
                        }),
                        _ => None,
                    };

                    if let Some(iface_name) = interface_name {
                        let method_name = field.to_string();
                        let iface = match self.context.get_interface(&iface_name) {
                            Some(i) => i,
                            None => {
                                self.record(TypeErrorKind::UndefinedType(iface_name));
                                return HirType::Unknown;
                            }
                        };

                        let method = iface.methods.and_then(|methods| {
                            methods
                                .iter()
                                .find(|m| m.unmangled_name.to_string() == method_name)
                        });
                        let Some(method) = method else {
                            self.record(TypeErrorKind::Generic(format!(
                                "no method `{}` on interface `{}`",
                                method_name, iface_name
                            )));
                            return HirType::Unknown;
                        };

                        let total_params = method.params.map(|p| p.len()).unwrap_or(0);
                        let expected_args = total_params.saturating_sub(1);
                        if args.len() != expected_args {
                            self.record(TypeErrorKind::InvalidFunctionCall {
                                expected_args,
                                found_args: args.len(),
                            });
                        }

                        if let Some(params) = method.params {
                            if let Some(HirParam::This {
                                kind, multi_place, ..
                            }) = params.first()
                            {
                                let requires_mut = match kind {
                                    ThisPassingKind::RefMut
                                    | ThisPassingKind::MutSafePtr
                                    | ThisPassingKind::MoveMut => true,
                                    ThisPassingKind::MultiPlace => multi_place.is_some_and(|a| {
                                        a.iter().any(|x| x.ref_kind == RefKind::Unique)
                                    }),
                                    _ => false,
                                };
                                if requires_mut {
                                    self.recover(
                                        self.check_receiver_is_mutable(object, field.as_str()),
                                        (),
                                    );
                                }

                                if let (ThisPassingKind::MultiPlace, Some(accesses)) =
                                    (kind, multi_place)
                                {
                                    let loans = self.register_multi_place_loans(object, accesses);
                                    for loan in loans {
                                        self.borrow_checker.end_loan_now(loan);
                                    }
                                } else if matches!(
                                    kind,
                                    ThisPassingKind::Move | ThisPassingKind::MoveMut
                                ) {
                                    self.check_and_record_value_use(object, &obj_type);
                                } else if let Some(place) = self.resolve_place(object) {
                                    let borrow_kind = if requires_mut {
                                        BorrowKind::Mutable
                                    } else {
                                        BorrowKind::Shared
                                    };
                                    self.check_borrow_use(object, place, borrow_kind);
                                }
                            }
                            for (arg, param) in args.iter().zip(params.iter().skip(1)) {
                                let arg_type = self.check_expr(arg);
                                self.check_and_record_value_use(arg, &arg_type);
                                if let Some(param_type) = param.get_type() {
                                    let result = self.types_compatible(param_type, &arg_type);
                                    self.recover(result, ());
                                }
                            }
                        }

                        return method.return_type.unwrap_or(HirType::Void);
                    }

                    let (struct_name_id, type_name, func) =
                        match self.resolve_callable_method(stripped, &field.to_string()) {
                            Some(found) => found,
                            None => {
                                self.record(TypeErrorKind::Generic(format!(
                                    "no method `{}` on `{}`",
                                    field,
                                    self.type_to_string(&obj_type)
                                )));
                                return HirType::Unknown;
                            }
                        };
                    self.check_unsafe_call(&func, &format!("{}.{}", type_name, field));

                    let total_params = func.params.map(|p| p.len()).unwrap_or(0);
                    let expected_args = total_params.saturating_sub(1);
                    if args.len() != expected_args {
                        self.record(TypeErrorKind::InvalidFunctionCall {
                            expected_args,
                            found_args: args.len(),
                        });
                    }

                    let struct_type_args: &[HirType<'a, 'bump>] = match stripped {
                        HirType::Struct { type_args, .. } => type_args,
                        _ => &[],
                    };
                    let method_subs =
                        self.generic_substitutions_for_struct(struct_name_id, struct_type_args);

                    let unsubstituted_ret_ty = func.return_type.unwrap_or(HirType::Void);
                    let ret_ty = if method_subs.is_empty() {
                        unsubstituted_ret_ty
                    } else {
                        self.substitute_type_local(&unsubstituted_ret_ty, &method_subs)
                    };
                    self.record_method_occurrence(*span, *field, ret_ty, struct_name_id);

                    let template = if self.return_type_may_alias(&ret_ty) {
                        Some(self.analyze_ref_template(&func))
                    } else {
                        None
                    };

                    if let Some(params) = func.params {
                        let mut receiver_multi_place_loans: Vec<LoanId> = Vec::new();

                        if let Some(HirParam::This {
                            kind,
                            multi_place,
                            span: _,
                        }) = params.first()
                        {
                            let requires_mut = match kind {
                                ThisPassingKind::RefMut
                                | ThisPassingKind::MutSafePtr
                                | ThisPassingKind::MoveMut => true,
                                ThisPassingKind::MultiPlace => multi_place.is_some_and(|a| {
                                    a.iter().any(|x| x.ref_kind == RefKind::Unique)
                                }),
                                _ => false,
                            };
                            if requires_mut {
                                self.recover(
                                    self.check_receiver_is_mutable(object, field.as_str()),
                                    (),
                                );
                            }

                            if let (ThisPassingKind::MultiPlace, Some(accesses)) =
                                (kind, multi_place)
                            {
                                receiver_multi_place_loans =
                                    self.register_multi_place_loans(object, accesses);
                            } else {
                                let has_precise_template =
                                    matches!(template, Some(RefTemplate::Path { .. }));

                                if matches!(kind, ThisPassingKind::Move | ThisPassingKind::MoveMut)
                                {
                                    self.check_and_record_value_use(object, &obj_type);
                                } else if !has_precise_template {
                                    if let Some(place) = self.resolve_place(object) {
                                        let borrow_kind = if requires_mut {
                                            BorrowKind::Mutable
                                        } else {
                                            BorrowKind::Shared
                                        };

                                        if !requires_mut && !self.return_type_may_alias(&ret_ty) {
                                            self.check_borrow_use_shell(expr, place, borrow_kind);
                                        } else {
                                            self.check_borrow_use(expr, place, borrow_kind);
                                        }
                                    }
                                }
                            }

                            // defer entirely to finalize_call_loans below, which checks
                            // the precise resolved place (such as ptr[Const(1)] vs
                            // ptr[Const(2)]) and can prove index-disjointness that the
                            // whole-receiver check can't.
                        }

                        let method_params: &[HirParam<'a, 'bump>] = if method_subs.is_empty() {
                            params
                        } else {
                            self.substitute_params_local(params, &method_subs)
                        };
                        let normal_params: &[HirParam<'a, 'bump>] =
                            method_params.get(1..).unwrap_or(&[]);

                        let read_templates = self.analyze_read_templates(&func);
                        for (arg_idx, arg) in args.iter().enumerate() {
                            if let HirExpr::Ref {
                                expr: inner,
                                ref_kind: RefKind::Shared,
                                ..
                            } = arg
                            {
                                if let Some(rt) = read_templates.get(arg_idx) {
                                    self.check_call_arg_read_effects(inner, rt, args);
                                }
                            }
                        }

                        let arg_loans =
                            self.check_all_func_args(args, normal_params, None, Some(func));

                        if let Some(loan_id) = self.finalize_call_loans(
                            Some(object),
                            args,
                            arg_loans,
                            &ret_ty,
                            template,
                        ) {
                            self.call_loans.insert(Self::expr_key(expr), loan_id);
                            for loan in receiver_multi_place_loans {
                                self.borrow_checker.end_loan_now(loan);
                            }
                        }
                    }

                    ret_ty
                }
                HirExpr::ModuleAccess(access) => {
                    self.set_span(access.span);
                    let member_name = access.member.to_string();

                    let is_named_import = if access.path.len() == 1 {
                        self.imports_by_module
                            .get(&self.context.current_module_idx)
                            .map(|imp| imp.named.contains_key(&access.path[0]))
                            .unwrap_or(false)
                    } else {
                        false
                    };

                    let alias_module_idx: Option<usize> = if access.path.len() == 1 {
                        self.imports_by_module
                            .get(&self.context.current_module_idx)
                            .and_then(|imp| {
                                imp.named
                                    .get(&access.path[0])
                                    .or_else(|| imp.module_aliases.get(&access.path[0]))
                            })
                            .copied()
                    } else {
                        None
                    };

                    let (resolved_module_idx, assoc_type_name): (Option<usize>, Option<StrId>) =
                        if let Some(midx) = alias_module_idx {
                            if is_named_import {
                                (Some(midx), Some(access.path[0]))
                            } else {
                                (Some(midx), None)
                            }
                        } else {
                            match self
                                .context
                                .dep_graph
                                .borrow()
                                .resolve_module_path(access.path)
                            {
                                Some(midx) => (Some(midx), None),
                                None => match access.path.split_last() {
                                    Some((&type_seg, module_path)) => {
                                        let midx = self
                                            .context
                                            .dep_graph
                                            .borrow()
                                            .resolve_module_path(module_path);
                                        (midx, midx.map(|_| type_seg))
                                    }
                                    None => (None, None),
                                },
                            }
                        };

                    let free_func = resolved_module_idx
                        .and_then(|midx| self.context.get_module_function(midx, &member_name));

                    let mangled_type_name: Option<String> = assoc_type_name.and_then(|t| {
                        let midx = resolved_module_idx?;
                        Some(
                            self.context
                                .dep_graph
                                .borrow()
                                .mangle_type_name(midx, t, &self.context.string_pool)
                                .to_string(),
                        )
                    });

                    let method_func = if free_func.is_none() {
                        mangled_type_name
                            .or_else(|| access.path.last().map(|s| s.to_string()))
                            .and_then(|tn| self.context.get_method(&tn, &member_name).copied())
                    } else {
                        None
                    };

                    let func = match free_func.or(method_func) {
                        Some(f) => f,
                        None => {
                            let path_str = access
                                .path
                                .iter()
                                .map(|s| s.to_string())
                                .collect::<Vec<_>>()
                                .join("::");
                            let qualified_name = format!("{}::{}", path_str, member_name);
                            let candidate_modules = self
                                .context
                                .dep_graph
                                .borrow()
                                .find_function_by_name_anywhere(access.member);
                            if candidate_modules.is_empty() {
                                self.record(TypeErrorKind::UndefinedFunction(qualified_name));
                            } else {
                                let suggestion_paths: Vec<String> = candidate_modules
                                    .iter()
                                    .filter_map(|&midx| {
                                        self.context.dep_graph.borrow().get_module_package(midx)
                                    })
                                    .map(|pkg| pkg.to_string())
                                    .collect();
                                self.record(TypeErrorKind::UndefinedFunctionWithSuggestion {
                                    name: qualified_name,
                                    suggested_modules: suggestion_paths,
                                });
                            }

                            return HirType::Unknown;
                        }
                    };

                    self.check_module_path_imported(access.path);

                    let expected_args = func.params.map(|p| p.len()).unwrap_or(0);
                    if args.len() != expected_args {
                        self.record(TypeErrorKind::InvalidFunctionCall {
                            expected_args,
                            found_args: args.len(),
                        });
                    }
                    if let Some(params) = func.params {
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

                        let arg_loans = self.check_all_func_args(args, params, None, Some(func));

                        let ret_ty = func.return_type.unwrap_or(HirType::Void);
                        if !self.return_type_may_alias(&ret_ty) {
                            for loan in arg_loans {
                                self.borrow_checker.end_loan_now(loan);
                            }
                        }
                    }
                    let ret_ty = func.return_type.unwrap_or(HirType::Void);
                    if let Some(midx) = resolved_module_idx {
                        if free_func.is_some() {
                            self.record_item_occurrence(access.span, access.member, ret_ty, midx);
                        } else if let Some(target_type) = assoc_type_name {
                            self.record_method_occurrence(
                                access.span,
                                access.member,
                                ret_ty,
                                target_type,
                            );
                        }
                    }
                    ret_ty
                }

                other => {
                    self.set_span(*span);
                    let callee_type = self.check_expr(other);
                    match callee_type {
                        HirType::Lambda { return_type, .. } => *return_type,
                        _ => {
                            self.record(TypeErrorKind::Generic(format!(
                                "Expression of type `{}` is not callable",
                                self.type_to_string(&callee_type)
                            )));
                            HirType::Unknown
                        }
                    }
                }
            },
            HirExpr::FieldAccess {
                object,
                field,
                span,
            } => {
                self.set_span(*span);
                self.check_field_access(object, *field)
            }
            HirExpr::StructInit {
                name,
                args,
                span,
                type_args,
            } => {
                self.set_span(*span);
                let HirExpr::Ident(struct_name_id, name_span) = name else {
                    return HirType::Void;
                };
                let struct_name_str = self.str_id_to_string(*struct_name_id);
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

                let resolved_field_types: Vec<HirType<'a, 'bump>> = match (
                    is_generic_decl,
                    type_args,
                ) {
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

                for field_init in *args {
                    let field_name_str = self.str_id_to_string(field_init.name);
                    let field_idx = ty_struct
                        .fields
                        .iter()
                        .position(|f| self.str_id_to_string(f.name) == field_name_str);

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
            HirExpr::InterfaceCall {
                callee,
                interface,
                args,
                span: _,
            } => {
                let (object, field) = match callee {
                    HirExpr::FieldAccess { object, field, .. }
                    | HirExpr::Get { object, field, .. } => (object, field),
                    other => {
                        self.record(TypeErrorKind::Generic(format!(
                            "interface call callee must be a field access, found {:?}",
                            other
                        )));
                        return HirType::Unknown;
                    }
                };
                let _ = self.check_expr(object);

                let iface_name = interface.to_string();
                let Some(iface) = self.context.get_interface(&iface_name) else {
                    self.record(TypeErrorKind::UndefinedType(iface_name));
                    return HirType::Unknown;
                };

                let method_name = field.to_string();
                let Some(method) = iface.methods.and_then(|methods| {
                    methods
                        .iter()
                        .find(|m| m.unmangled_name.to_string() == method_name)
                }) else {
                    self.record(TypeErrorKind::Generic(format!(
                        "no method `{}` on interface `{}`",
                        method_name, iface_name
                    )));
                    return HirType::Unknown;
                };

                let total_params = method.params.map(|p| p.len()).unwrap_or(0);
                let expected_args = total_params.saturating_sub(1); // exclude `this`
                if args.len() != expected_args {
                    self.record(TypeErrorKind::InvalidFunctionCall {
                        expected_args,
                        found_args: args.len(),
                    });
                }

                if let Some(params) = method.params {
                    for (arg, param) in args.iter().zip(params.iter().skip(1)) {
                        let arg_type = match param.get_type() {
                            Some(pt) => self.check_expr_expected(arg, pt),
                            None => self.check_expr(arg),
                        };
                        self.check_and_record_value_use(arg, &arg_type);
                        if let Some(pt) = param.get_type() {
                            self.recover(self.types_compatible(pt, &arg_type), ());
                        }
                    }
                }

                method.return_type.unwrap_or(HirType::Void)
            }

            HirExpr::Assignment {
                target,
                op,
                value,
                span,
            } => {
                self.set_span(*span);

                if let HirExpr::FieldAccess { object, field, .. }
                | HirExpr::Get { object, field, .. } = target
                {
                    let field_name = self.str_id_to_string(*field);
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
                                    self.type_to_string(&value_type)
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
                    if is_uninit_value {
                        self.borrow_checker.mark_place_uninit(place);
                    } else {
                        self.borrow_checker.mark_place_init(place);
                    }
                }

                if let HirExpr::Ident(name, _) = target {
                    let var_name = self.str_id_to_string(*name);
                    if self.context.is_local_binding(&var_name)
                        && !self.context.is_mutable(&var_name)
                    {
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
                    HirExpr::FieldAccess { object, field, .. }
                    | HirExpr::Get { object, field, .. } => {
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
                                    if !matches!(
                                        target_node,
                                        InitNode::Whole(InitStatus::Initialized)
                                    ) {
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
            HirExpr::InterpolatedString(parts) => {
                for part in *parts {
                    if let ir::hir::InterpolationPart::Expr(e) = part {
                        self.check_expr(e);
                    }
                }
                HirType::String
            }
            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
            } => self.check_enum_init(expr, enum_name, variant, args, type_args, *span, None),
            HirExpr::ExprList { list, span } => {
                self.set_span(*span);
                let mut last = HirType::Void;
                for e in *list {
                    last = self.check_expr(e);
                }
                last
            }
            HirExpr::Get {
                object,
                field,
                span,
            } => {
                self.set_span(*span);
                self.check_field_access(object, *field)
            }
            HirExpr::Comparison {
                left,
                op,
                right,
                span,
            } => {
                self.set_span(*span);
                let left_type = self.check_expr(left);
                let right_type = self.check_expr(right);
                let result = self.check_binary_op(&left_type, op, &right_type);
                self.recover(result, HirType::Unknown)
            }
            HirExpr::Deref { expr, span } => {
                self.set_span(*span);
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
                            self.type_to_string(&inner_ty)
                        )));
                        HirType::Unknown
                    }
                }
            }
            HirExpr::Ref {
                expr,
                ref_kind,
                span,
            } => {
                let ty = self.check_ref_expr(expr, *ref_kind, *span, true);
                if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                    self.optimistically_mark_mut_target_init(expr);
                }
                ty
            }
            HirExpr::This { span } => {
                self.set_span(*span);
                match self.context.get_variable("this") {
                    Some((symbol_id, ty)) => {
                        self.occurrences.push((
                            *span,
                            self.this_id,
                            ty,
                            self.context.current_module_idx,
                            symbol_id,
                            false,
                        ));
                        ty
                    }
                    None => {
                        self.record(TypeErrorKind::Generic(
                            "`this` used outside of a method that has a `this` receiver"
                                .to_string(),
                        ));
                        HirType::Unknown
                    }
                }
            }
            HirExpr::ModuleAccess(access) => {
                let member_name = access.member.to_string();

                let is_named_import = if access.path.len() == 1 {
                    self.imports_by_module
                        .get(&self.context.current_module_idx)
                        .map(|imp| imp.named.contains_key(&access.path[0]))
                        .unwrap_or(false)
                } else {
                    false
                };

                let alias_module_idx: Option<usize> = if access.path.len() == 1 {
                    self.imports_by_module
                        .get(&self.context.current_module_idx)
                        .and_then(|imp| {
                            imp.named
                                .get(&access.path[0])
                                .or_else(|| imp.module_aliases.get(&access.path[0]))
                        })
                        .copied()
                } else {
                    None
                };

                let (resolved_module_idx, assoc_type_name): (Option<usize>, Option<StrId>) =
                    if let Some(midx) = alias_module_idx {
                        if is_named_import {
                            (Some(midx), Some(access.path[0]))
                        } else {
                            (Some(midx), None)
                        }
                    } else {
                        match self
                            .context
                            .dep_graph
                            .borrow()
                            .resolve_module_path(access.path)
                        {
                            Some(midx) => (Some(midx), None),
                            None => match access.path.split_last() {
                                Some((&type_seg, module_path)) => {
                                    let midx = self
                                        .context
                                        .dep_graph
                                        .borrow()
                                        .resolve_module_path(module_path);
                                    (midx, midx.map(|_| type_seg))
                                }
                                None => (None, None),
                            },
                        }
                    };

                let free_func = resolved_module_idx
                    .and_then(|midx| self.context.get_module_function(midx, &member_name));

                let mangled_type_name: Option<String> = assoc_type_name.and_then(|t| {
                    let midx = resolved_module_idx?;
                    let pkg = self.context.dep_graph.borrow().get_module_package(midx)?;
                    Some(format!("{}_{}", pkg.to_string(), t.to_string()))
                });

                let method_func = if free_func.is_none() {
                    mangled_type_name
                        .or_else(|| access.path.last().map(|s| s.to_string()))
                        .and_then(|tn| self.context.get_method(&tn, &member_name).copied())
                } else {
                    None
                };

                let func = match free_func.or(method_func) {
                    Some(f) => f,
                    None => {
                        let path_str = access
                            .path
                            .iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<_>>()
                            .join("::");
                        let qualified_name = format!("{}.{}", path_str, member_name);
                        let candidate_modules = self
                            .context
                            .dep_graph
                            .borrow()
                            .find_function_by_name_anywhere(access.member);
                        if candidate_modules.is_empty() {
                            self.record(TypeErrorKind::UndefinedFunction(qualified_name));
                        } else {
                            let suggestion_paths: Vec<String> = candidate_modules
                                .iter()
                                .filter_map(|&midx| {
                                    self.context.dep_graph.borrow().get_module_package(midx)
                                })
                                .map(|pkg| pkg.to_string())
                                .collect();
                            self.record(TypeErrorKind::UndefinedFunctionWithSuggestion {
                                name: qualified_name,
                                suggested_modules: suggestion_paths,
                            });
                        }

                        return HirType::Unknown;
                    }
                };

                self.check_unsafe_call(&func, &member_name);

                self.check_module_path_imported(access.path);

                let param_types: Vec<HirType<'a, 'bump>> = func
                    .params
                    .unwrap_or(&[])
                    .iter()
                    .filter_map(|p| p.get_type().copied())
                    .collect();
                HirType::Lambda {
                    params: self.context.bump.alloc_slice(&param_types),
                    return_type: self
                        .context
                        .bump
                        .alloc_value(func.return_type.unwrap_or(HirType::Void)),
                }
            }
            HirExpr::Lambda {
                params,
                return_type,
                body,
                span,
                ..
            } => {
                self.set_span(*span);
                let mut lambda_context = self.context.create_child_scope();
                for p in *params {
                    let param_name = self.str_id_to_string(p.name);
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
            HirExpr::Index {
                object,
                index,
                span,
            } => {
                self.set_span(*span);

                let object_ty = self.check_expr_suppressed(object);
                let index_ty = self.check_expr(index);

                self.recover(self.types_compatible(&HirType::I64, &index_ty), ());

                if let Some((root, path)) = self.static_field_path(object) {
                    if let Some(node) = self.init_state.get(&root).cloned() {
                        let target = Self::node_at_path_ref(&node, &path);
                        match index {
                            HirExpr::Number(i, _) => {
                                let covered = match target {
                                    InitNode::Array { ranges, .. } => {
                                        ranges.contains_range(*i, *i + 1)
                                    }
                                    InitNode::Whole(InitStatus::Initialized) => true,
                                    _ => false,
                                };
                                if !covered {
                                    let root_str = self.str_id_to_string(root);
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
                                    let root_str = self.str_id_to_string(root);
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
                                self.type_to_string(&object_ty)
                            )));
                            HirType::Unknown
                        }
                    },
                }
            }
            HirExpr::ArrayLiteral { elements, span } => {
                self.set_span(*span);

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
            HirExpr::GenericIdent(..) => todo!(),
            HirExpr::Cast {
                expr,
                target_type,
                span,
            } => {
                self.set_span(*span);
                let source_type = self.check_expr(expr);
                self.check_and_record_value_use(expr, &source_type);
                let result = self.check_cast_legality(&source_type, target_type);
                self.recover(result, ());
                *target_type
            }
            HirExpr::Char(_, _) => HirType::Char,
            HirExpr::UnknownIntrinsic { span, name } => {
                self.recover(
                    Err(TypeErrorKind::Generic(format!("Unknown intrinsic {}", name)).at(*span)),
                    (),
                );
                HirType::Unknown
            }
        }
    }

    fn join_value_types(&mut self, branches: &[HirType<'a, 'bump>]) -> HirType<'a, 'bump> {
        let mut result: Option<HirType<'a, 'bump>> = None;
        for ty in branches {
            if matches!(ty, HirType::Never) {
                continue;
            }
            match result {
                None => result = Some(*ty),
                Some(expected) => {
                    let check = self.types_compatible(&expected, ty);
                    self.recover(check, ());
                }
            }
        }
        result.unwrap_or(HirType::Never)
    }

    fn check_cast_legality(
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
                self.type_to_string(source),
                self.type_to_string(target),
            ))
            .at(self.current_span))
        }
    }

    fn infer_provenance(&self, expr: &HirExpr<'a, 'bump>) -> Option<ProvenanceAnnotation<'bump>> {
        let mut segments = Vec::new();
        let root = self.infer_provenance_root(expr, &mut segments)?;
        segments.reverse();
        Some(ProvenanceAnnotation {
            root,
            path: self.context.bump.alloc_slice(&segments),
        })
    }

    fn infer_provenance_root(
        &self,
        expr: &HirExpr<'a, 'bump>,
        segments: &mut Vec<ProvenancePathSegment>,
    ) -> Option<ProvenanceRoot> {
        match expr {
            HirExpr::Ident(name, _) => Some(ProvenanceRoot::Var(*name)),
            HirExpr::This { .. } => Some(ProvenanceRoot::ThisRoot),

            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                segments.push(ProvenancePathSegment::Field(*field));
                self.infer_provenance_root(object, segments)
            }

            HirExpr::Deref { expr: inner, .. } => {
                segments.push(ProvenancePathSegment::Deref);
                self.infer_provenance_root(inner, segments)
            }

            HirExpr::Index { object, .. } => self.infer_provenance_root(object, segments),

            HirExpr::ModuleAccess(access) => {
                let module_idx = self
                    .context
                    .dep_graph
                    .borrow()
                    .resolve_module_path(access.path)?;
                self.context
                    .dep_graph
                    .borrow()
                    .resolve_global_const(module_idx, access.member)?;
                Some(ProvenanceRoot::Global {
                    module_idx,
                    name: access.member,
                })
            }

            _ => None,
        }
    }

    fn check_ref_expr(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
        register_loan: bool,
    ) -> HirType<'a, 'bump> {
        self.set_span(span);

        let inner_ty = if ref_kind != RefKind::Shared {
            self.check_expr_as_place(expr)
        } else {
            self.check_expr(expr)
        };

        let provenance = self.infer_provenance(expr);

        if register_loan {
            if let Some(place) = self.resolve_place(expr) {
                let result = match ref_kind {
                    RefKind::Unique => self.borrow_checker.borrow_mut(place),
                    RefKind::Alias => self.borrow_checker.borrow_alias(place),
                    RefKind::Shared => self.borrow_checker.borrow_shared(place),
                };
                if let Err(e) = result {
                    let msg = self.describe_borrow_error(&e, provenance.as_ref());
                    self.record(TypeErrorKind::Generic(msg));
                }
            }
        }

        HirType::Ref {
            inner: self.context.bump.alloc_value(inner_ty),
            ref_kind,
            provenance,
        }
    }

    fn check_pattern_against_type(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
    ) {
        if let HirType::Nullable(inner) = scrutinee_ty {
            if !matches!(pattern, HirPattern::Null) {
                let inner_ty = **inner;
                return self.check_pattern_against_type(pattern, &inner_ty);
            }
        }
        match pattern {
            HirPattern::Null => {
                if !matches!(scrutinee_ty, HirType::Nullable(_)) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: "null".to_string(),
                    });
                }
            }
            HirPattern::EnumVariant {
                variant, bindings, ..
            } => {
                let HirType::Enum {
                    name: enum_name, ..
                } = scrutinee_ty
                else {
                    return;
                };
                let enum_name_str = self.str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == *variant) else {
                    self.record(TypeErrorKind::Generic(format!(
                        "enum `{}` has no variant `{}`",
                        enum_name_str, variant
                    )));
                    return;
                };
                if bindings.len() != variant_def.fields.len() {
                    self.record(TypeErrorKind::Generic(format!(
                        "variant `{}::{}` has {} field(s), but the pattern binds {}",
                        enum_name_str,
                        variant,
                        variant_def.fields.len(),
                        bindings.len()
                    )));
                }
            }
            HirPattern::Boolean(_) => {
                if !matches!(scrutinee_ty, HirType::Boolean) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: "bool".to_string(),
                    });
                }
            }
            HirPattern::Number(_) => {
                if !self.is_integer(scrutinee_ty) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: "integer".to_string(),
                    });
                }
            }
            HirPattern::String(_) => {
                if !matches!(scrutinee_ty, HirType::String) {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: "str".to_string(),
                    });
                }
            }

            HirPattern::Ident(_) | HirPattern::Wildcard => {}

            HirPattern::Tuple(patterns) => match scrutinee_ty {
                HirType::Tuple(elems) => {
                    if patterns.len() != elems.len() {
                        self.record(TypeErrorKind::Generic(format!(
                            "tuple pattern has {} element(s), but the scrutinee has {}",
                            patterns.len(),
                            elems.len()
                        )));
                    }
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.check_pattern_against_type(sub_pattern, elem_ty);
                    }
                }
                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: format!("tuple pattern with {} element(s)", patterns.len()),
                    });
                }
            },

            HirPattern::Array(patterns) => match scrutinee_ty {
                HirType::Array(inner, len) => {
                    if patterns.len() != *len {
                        self.record(TypeErrorKind::Generic(format!(
                            "array pattern has {} element(s), but the array type `{}` has length {}",
                            patterns.len(),
                            self.type_to_string(scrutinee_ty),
                            len
                        )));
                    }
                    for sub_pattern in patterns.iter() {
                        self.check_pattern_against_type(sub_pattern, inner);
                    }
                }
                HirType::Slice(inner) => {
                    for sub_pattern in patterns.iter() {
                        self.check_pattern_against_type(sub_pattern, inner);
                    }
                }
                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: format!("array pattern with {} element(s)", patterns.len()),
                    });
                }
            },

            HirPattern::Struct { name, fields } => match scrutinee_ty {
                HirType::Enum {
                    name: enum_name, ..
                } => {
                    let enum_name_str = self.str_id_to_string(*enum_name);
                    let Some(def) = self.context.get_enum(&enum_name_str) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == *name) else {
                        self.record(TypeErrorKind::Generic(format!(
                            "enum `{}` has no variant `{}`",
                            enum_name_str, name
                        )));
                        return;
                    };

                    let mut seen: std::collections::HashSet<StrId> =
                        std::collections::HashSet::new();
                    for (field_name, sub_pattern) in fields.iter() {
                        if !seen.insert(*field_name) {
                            self.record(TypeErrorKind::Generic(format!(
                                "field `{}` matched more than once in this pattern",
                                self.str_id_to_string(*field_name)
                            )));
                            continue;
                        }
                        let Some(field_def) =
                            variant_def.fields.iter().find(|f| f.name == *field_name)
                        else {
                            self.record(TypeErrorKind::Generic(format!(
                                "variant `{}::{}` has no field `{}`",
                                enum_name_str,
                                name,
                                self.str_id_to_string(*field_name)
                            )));
                            continue;
                        };
                        self.check_pattern_against_type(sub_pattern, &field_def.field_type);
                    }

                    let missing: Vec<&str> = variant_def
                        .fields
                        .iter()
                        .filter(|f| !fields.iter().any(|(fname, _)| fname == &f.name))
                        .map(|f| f.name.as_str())
                        .collect();
                    if !missing.is_empty() {
                        self.record(TypeErrorKind::Generic(format!(
                            "pattern doesn't bind field(s) {} of variant `{}::{}`",
                            missing.join(", "),
                            enum_name_str,
                            name
                        )));
                    }
                }

                HirType::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let struct_name_str = self.str_id_to_string(*struct_name);
                    let Some(def) = self.context.get_struct(&struct_name_str) else {
                        return;
                    };

                    let mut seen: std::collections::HashSet<StrId> =
                        std::collections::HashSet::new();
                    for (field_name, sub_pattern) in fields.iter() {
                        if !seen.insert(*field_name) {
                            self.record(TypeErrorKind::Generic(format!(
                                "field `{}` matched more than once in this pattern",
                                self.str_id_to_string(*field_name)
                            )));
                            continue;
                        }
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            self.record(TypeErrorKind::FieldNotFound {
                                struct_name: struct_name_str.clone(),
                                field: self.str_id_to_string(*field_name),
                            });
                            continue;
                        };
                        let field_ty = field_types
                            .get(field_idx)
                            .copied()
                            .unwrap_or(def.fields[field_idx].field_type);
                        self.check_pattern_against_type(sub_pattern, &field_ty);
                    }
                }

                _ => {
                    self.record(TypeErrorKind::TypeMismatch {
                        expected: self.type_to_string(scrutinee_ty),
                        found: format!("named-field pattern `{}`", name),
                    });
                }
            },

            HirPattern::Or(patterns) => {
                if patterns.is_empty() {
                    self.record(TypeErrorKind::Generic(
                        "or-pattern must have at least one alternative".to_string(),
                    ));
                    return;
                }

                for sub_pattern in patterns.iter() {
                    self.check_pattern_against_type(sub_pattern, scrutinee_ty);
                }

                let mut first_bindings: Option<Vec<(StrId, HirType<'a, 'bump>)>> = None;
                for sub_pattern in patterns.iter() {
                    let mut bindings = Vec::new();
                    self.collect_pattern_bindings(sub_pattern, scrutinee_ty, &mut bindings);
                    bindings.sort_by(|(na, _), (nb, _)| {
                        self.str_id_to_string(*na).cmp(&self.str_id_to_string(*nb))
                    });

                    match &first_bindings {
                        None => first_bindings = Some(bindings),
                        Some(expected) => {
                            if !self.bindings_match(expected, &bindings) {
                                let names = |v: &[(StrId, HirType<'a, 'bump>)]| {
                                    v.iter()
                                        .map(|(n, _)| self.str_id_to_string(*n))
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                };
                                self.record(TypeErrorKind::Generic(format!(
                                    "all alternatives of an or-pattern must bind the same names with the same types: \
                                     found `{}` in one alternative but `{}` in another",
                                    names(expected),
                                    names(&bindings),
                                )));
                            }
                        }
                    }
                }
            }
        }
    }

    fn resolve_callable_method(
        &self,
        ty: &HirType<'a, 'bump>,
        method_name: &str,
    ) -> Option<(StrId, String, HirFunc<'a, 'bump>)> {
        let try_name = |n: String| -> Option<(StrId, String, HirFunc<'a, 'bump>)> {
            let id = StrId(self.context.string_pool.intern(&n));
            self.context
                .get_method(&n, method_name)
                .map(|f| (id, n, *f))
        };

        match ty {
            HirType::Struct { name, .. } => {
                let struct_name_str = name.to_string();
                if let Some(hit) = try_name(struct_name_str.clone()) {
                    return Some(hit);
                }
                self.resolve_default_interface_method(*name, &struct_name_str, method_name)
            }
            HirType::Slice(elem) | HirType::Array(elem, _) => {
                if let Some(elem_name) = self.builtin_element_name(elem) {
                    if let Some(hit) = try_name(format!("slice_{}", elem_name)) {
                        return Some(hit);
                    }
                }
                try_name("slice".to_string())
            }
            other => try_name(self.builtin_element_name(other)?),
        }
    }

    fn resolve_default_interface_method(
        &self,
        struct_name: StrId,
        struct_name_str: &str,
        method_name: &str,
    ) -> Option<(StrId, String, HirFunc<'a, 'bump>)> {
        let interfaces = self.context.struct_interfaces.get(struct_name_str)?;
        for iface_name in interfaces {
            let Some(iface) = self.context.get_interface(iface_name) else {
                continue;
            };
            let Some(methods) = iface.methods else {
                continue;
            };
            if let Some(m) = methods
                .iter()
                .find(|m| m.unmangled_name.as_str() == method_name && m.body.is_some())
            {
                return Some((struct_name, struct_name_str.to_string(), *m));
            }
        }
        None
    }

    fn builtin_element_name(&self, ty: &HirType<'a, 'bump>) -> Option<String> {
        Some(match ty {
            HirType::I8 => "i8".into(),
            HirType::I16 => "i16".into(),
            HirType::I32 => "i32".into(),
            HirType::I64 => "i64".into(),
            HirType::I128 => "i128".into(),
            HirType::U8 => "u8".into(),
            HirType::U16 => "u16".into(),
            HirType::U32 => "u32".into(),
            HirType::U64 => "u64".into(),
            HirType::U128 => "u128".into(),
            HirType::Usize => "usize".into(),
            HirType::Isize => "isize".into(),
            HirType::F32 => "f32".into(),
            HirType::F64 => "f64".into(),
            HirType::Boolean => "bool".into(),
            HirType::String => "str".into(),
            HirType::Char => "char".into(),
            HirType::Struct { name, .. } => name.to_string(),
            _ => return None,
        })
    }

    fn register_pattern_bindings(
        &mut self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        mode: BindingMode,
        scrutinee_provenance: Option<ProvenanceAnnotation<'bump>>,
        scrutinee_place: Option<PlaceId>,
    ) {
        if let HirType::Nullable(inner) = scrutinee_ty {
            if !matches!(pattern, HirPattern::Null) {
                let inner_ty = **inner;
                return self.register_pattern_bindings(
                    pattern,
                    &inner_ty,
                    mode,
                    scrutinee_provenance,
                    scrutinee_place,
                );
            }
        }

        match pattern {
            HirPattern::Ident(name) => {
                let var_name = self.str_id_to_string(*name);
                let symbol_id = self.mint_symbol_id();
                let bound_ty = self.bind_leaf_type(*scrutinee_ty, mode, scrutinee_provenance);
                self.context.add_variable(var_name, bound_ty, symbol_id);

                self.borrow_checker.declare_local(*name);

                match mode {
                    BindingMode::ByRef(rk) => {
                        self.local_ref_kind.insert(*name, rk);
                        if let Some(src_place) = scrutinee_place {
                            self.local_provenance_place.insert(*name, src_place);
                            let result = match rk {
                                RefKind::Unique => self.borrow_checker.borrow_mut(src_place),
                                RefKind::Alias => self.borrow_checker.borrow_alias(src_place),
                                RefKind::Shared => self.borrow_checker.borrow_shared(src_place),
                            };
                            match result {
                                Ok(loan_id) => {
                                    self.loan_owners.insert(loan_id, *name);
                                }
                                Err(e) => {
                                    let msg = self
                                        .describe_borrow_error(&e, scrutinee_provenance.as_ref());
                                    self.record(TypeErrorKind::Generic(msg));
                                }
                            }
                        }
                    }
                    BindingMode::ByValue => {
                        // owned
                    }
                }
            }

            HirPattern::EnumVariant {
                variant, bindings, ..
            } => {
                let HirType::Enum {
                    name: enum_name, ..
                } = scrutinee_ty
                else {
                    return;
                };
                let enum_name_str = self.str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == *variant) else {
                    return;
                };
                for (binding_name, field) in bindings.iter().zip(variant_def.fields.iter()) {
                    let var_name = self.str_id_to_string(*binding_name);
                    let symbol_id = self.mint_symbol_id();
                    let field_provenance =
                        Self::extend_provenance(scrutinee_provenance, field.name, &self.context);
                    let bound_ty = self.bind_leaf_type(field.field_type, mode, field_provenance);
                    self.context.add_variable(var_name, bound_ty, symbol_id);
                }
            }

            HirPattern::Tuple(patterns) => {
                if let HirType::Tuple(elems) = scrutinee_ty {
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.register_pattern_bindings(
                            sub_pattern,
                            elem_ty,
                            mode,
                            scrutinee_provenance,
                            scrutinee_place,
                        );
                    }
                }
            }

            HirPattern::Array(patterns) => {
                let elem_ty = match scrutinee_ty {
                    HirType::Array(inner, _) | HirType::Slice(inner) => Some(**inner),
                    _ => None,
                };
                if let Some(elem_ty) = elem_ty {
                    for sub_pattern in patterns.iter() {
                        self.register_pattern_bindings(
                            sub_pattern,
                            &elem_ty,
                            mode,
                            scrutinee_provenance,
                            scrutinee_place,
                        );
                    }
                }
            }

            HirPattern::Struct { name, fields } => match scrutinee_ty {
                HirType::Enum {
                    name: enum_name, ..
                } => {
                    let enum_name_str = self.str_id_to_string(*enum_name);
                    let Some(def) = self.context.get_enum(&enum_name_str) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == *name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_def) =
                            variant_def.fields.iter().find(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        let field_provenance = Self::extend_provenance(
                            scrutinee_provenance,
                            *field_name,
                            &self.context,
                        );
                        self.register_pattern_bindings(
                            sub_pattern,
                            &field_def.field_type,
                            mode,
                            field_provenance,
                            scrutinee_place,
                        );
                    }
                }
                HirType::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let struct_name_str = self.str_id_to_string(*struct_name);
                    let Some(def) = self.context.get_struct(&struct_name_str) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        let field_ty = field_types
                            .get(field_idx)
                            .copied()
                            .unwrap_or(def.fields[field_idx].field_type);
                        let field_provenance = Self::extend_provenance(
                            scrutinee_provenance,
                            *field_name,
                            &self.context,
                        );
                        self.register_pattern_bindings(
                            sub_pattern,
                            &field_ty,
                            mode,
                            field_provenance,
                            scrutinee_place,
                        );
                    }
                }
                _ => {}
            },

            HirPattern::Or(patterns) => {
                if let Some(first) = patterns.first() {
                    self.register_pattern_bindings(
                        first,
                        scrutinee_ty,
                        mode,
                        scrutinee_provenance,
                        scrutinee_place,
                    );
                }
            }

            _ => {}
        }
    }

    fn bind_leaf_type(
        &self,
        field_ty: HirType<'a, 'bump>,
        mode: BindingMode,
        provenance: Option<ProvenanceAnnotation<'bump>>,
    ) -> HirType<'a, 'bump> {
        match mode {
            BindingMode::ByValue => field_ty,
            BindingMode::ByRef(rk) => HirType::Ref {
                inner: self.context.bump.alloc_value(field_ty),
                ref_kind: rk,
                provenance,
            },
        }
    }

    fn extend_provenance(
        base: Option<ProvenanceAnnotation<'bump>>,
        field: StrId,
        ctx: &TypeContext<'a, 'bump>,
    ) -> Option<ProvenanceAnnotation<'bump>> {
        let base = base?;
        let mut path: Vec<ProvenancePathSegment> = base.path.to_vec();
        path.push(ProvenancePathSegment::Field(field));
        Some(ProvenanceAnnotation {
            root: base.root,
            path: ctx.bump.alloc_slice(&path),
        })
    }

    fn collect_pattern_bindings(
        &self,
        pattern: &HirPattern<'bump>,
        scrutinee_ty: &HirType<'a, 'bump>,
        out: &mut Vec<(StrId, HirType<'a, 'bump>)>,
    ) {
        if let HirType::Nullable(inner) = scrutinee_ty {
            if !matches!(pattern, HirPattern::Null) {
                let inner_ty = **inner;
                return self.collect_pattern_bindings(pattern, &inner_ty, out);
            }
        }

        match pattern {
            HirPattern::Ident(name) => {
                out.push((*name, *scrutinee_ty));
            }

            HirPattern::EnumVariant {
                variant, bindings, ..
            } => {
                let HirType::Enum {
                    name: enum_name, ..
                } = scrutinee_ty
                else {
                    return;
                };
                let enum_name_str = self.str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let Some(variant_def) = def.variants.iter().find(|v| v.name == *variant) else {
                    return;
                };
                for (binding_name, field) in bindings.iter().zip(variant_def.fields.iter()) {
                    out.push((*binding_name, field.field_type));
                }
            }

            HirPattern::Tuple(patterns) => {
                if let HirType::Tuple(elems) = scrutinee_ty {
                    for (sub_pattern, elem_ty) in patterns.iter().zip(elems.iter()) {
                        self.collect_pattern_bindings(sub_pattern, elem_ty, out);
                    }
                }
            }

            HirPattern::Array(patterns) => {
                let elem_ty = match scrutinee_ty {
                    HirType::Array(inner, _) | HirType::Slice(inner) => Some(**inner),
                    _ => None,
                };
                if let Some(elem_ty) = elem_ty {
                    for sub_pattern in patterns.iter() {
                        self.collect_pattern_bindings(sub_pattern, &elem_ty, out);
                    }
                }
            }

            HirPattern::Struct { name, fields } => match scrutinee_ty {
                HirType::Enum {
                    name: enum_name, ..
                } => {
                    let enum_name_str = self.str_id_to_string(*enum_name);
                    let Some(def) = self.context.get_enum(&enum_name_str) else {
                        return;
                    };
                    let Some(variant_def) = def.variants.iter().find(|v| v.name == *name) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_def) =
                            variant_def.fields.iter().find(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        self.collect_pattern_bindings(sub_pattern, &field_def.field_type, out);
                    }
                }
                HirType::Struct {
                    name: struct_name,
                    field_types,
                    ..
                } => {
                    let struct_name_str = self.str_id_to_string(*struct_name);
                    let Some(def) = self.context.get_struct(&struct_name_str) else {
                        return;
                    };
                    for (field_name, sub_pattern) in fields.iter() {
                        let Some(field_idx) = def.fields.iter().position(|f| f.name == *field_name)
                        else {
                            continue;
                        };
                        let field_ty = field_types
                            .get(field_idx)
                            .copied()
                            .unwrap_or(def.fields[field_idx].field_type);
                        self.collect_pattern_bindings(sub_pattern, &field_ty, out);
                    }
                }
                _ => {}
            },

            HirPattern::Or(patterns) => {
                if let Some(first) = patterns.first() {
                    self.collect_pattern_bindings(first, scrutinee_ty, out);
                }
            }

            HirPattern::Wildcard
            | HirPattern::Number(_)
            | HirPattern::String(_)
            | HirPattern::Boolean(_)
            | HirPattern::Null => {}
        }
    }

    fn bindings_match(
        &self,
        a: &[(StrId, HirType<'a, 'bump>)],
        b: &[(StrId, HirType<'a, 'bump>)],
    ) -> bool {
        if a.len() != b.len() {
            return false;
        }
        a.iter()
            .zip(b.iter())
            .all(|((na, ta), (nb, tb))| na == nb && self.types_structurally_equal(ta, tb))
    }

    fn check_potential_this_param_for_move(
        &mut self,
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
            self.finalize_call_loans(None, args, arg_loans, &ret_ty, template);

            return Some(ret_ty);
        }
        Some(ret_ty)
    }

    fn instantiate_enum(
        &self,
        name: StrId,
        args: &[HirType<'a, 'bump>],
    ) -> Option<&'bump [(StrId, &'bump [HirType<'a, 'bump>])]> {
        if let Some(cached) = self.context.get_enum_instantiation(name, args) {
            return Some(cached);
        }

        let name_str = self.str_id_to_string(name);
        let def = self.context.get_enum(&name_str)?;
        let generics = def.generics?;
        if generics.len() != args.len() {
            return None;
        }

        let mut subs = FxHashMap::default();
        for (param, arg) in generics.iter().zip(args.iter()) {
            subs.insert(param.name, *arg);
        }

        let mut resolved_variants = Vec::with_capacity(def.variants.len());
        for variant in def.variants.iter() {
            let field_types: Vec<_> = variant
                .fields
                .iter()
                .map(|f| self.substitute_type_local(&f.field_type, &subs))
                .collect();
            resolved_variants.push((
                variant.name,
                self.context.bump.alloc_slice_copy(&field_types),
            ));
        }
        let result = self.context.bump.alloc_slice_copy(&resolved_variants);

        self.context
            .cache_enum_instantiation(name, args.to_vec(), result);
        Some(result)
    }

    fn callee_ignores_initial_contents(
        write_template: Option<&[bool]>,
        read_template: Option<&[ReadTemplate]>,
        i: usize,
    ) -> bool {
        let definitely_written = write_template
            .and_then(|t| t.get(i))
            .copied()
            .unwrap_or(false);
        let never_reads_contents = read_template
            .and_then(|t| t.get(i))
            .is_some_and(|rt| !Self::read_template_touches_contents(rt));
        definitely_written || never_reads_contents
    }

    fn check_ref_expr_deferring_init(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        ref_kind: RefKind,
        span: SourceSpan<'a>,
        register_loan: bool,
        skip_init: bool,
    ) -> HirType<'a, 'bump> {
        let prev = self.skip_slice_init_check;
        // Only arm the flag for a top-level slice operand, so it can't be
        // picked up by an unrelated nested `&mut x[..]`.
        self.skip_slice_init_check = skip_init && matches!(expr, HirExpr::Slice { .. });
        let ty = self.check_ref_expr(expr, ref_kind, span, register_loan);
        self.skip_slice_init_check = prev;
        ty
    }

    fn check_all_func_args(
        &mut self,
        args: &[HirExpr<'a, 'bump>],
        params: &[HirParam<'a, 'bump>],
        templated_base_param: Option<usize>,
        callee: Option<HirFunc<'a, 'bump>>,
    ) -> Vec<LoanId> {
        let mut arg_loans: Vec<LoanId> = Vec::new();
        let write_template: Option<Vec<bool>> = callee.map(|f| self.analyze_definite_writes(&f));
        let read_template: Option<Vec<ReadTemplate>> =
            callee.map(|f| self.analyze_read_templates(&f));

        for (i, (arg, param)) in args.iter().zip(params.iter()).enumerate() {
            let param_type = param.get_type();

            if let (
                HirParam::Normal {
                    multi_place: Some(accesses),
                    ..
                },
                HirExpr::Ref {
                    expr,
                    ref_kind,
                    span,
                },
            ) = (param, arg)
            {
                let arg_type = self.check_ref_expr(expr, *ref_kind, *span, false);
                if let Some(pt) = param_type {
                    self.recover(self.types_compatible(pt, &arg_type), ());
                }
                if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                    self.optimistically_mark_mut_target_init(expr);
                }
                arg_loans.extend(self.register_multi_place_loans(expr, accesses));
                continue;
            }

            if Some(i) == templated_base_param {
                if let HirExpr::Ref {
                    expr,
                    ref_kind,
                    span,
                } = arg
                {
                    let ignores_init = *ref_kind != RefKind::Shared
                        && Self::callee_ignores_initial_contents(
                            write_template.as_deref(),
                            read_template.as_deref(),
                            i,
                        );
                    let arg_type = self.check_ref_expr_deferring_init(
                        expr,
                        *ref_kind,
                        *span,
                        false,
                        ignores_init,
                    );
                    if let Some(pt) = param_type {
                        self.recover(self.types_compatible(pt, &arg_type), ());
                    }
                    if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                        self.optimistically_mark_mut_target_init(expr);
                    }
                } else {
                    let arg_type = match param_type {
                        Some(pt) => self.check_expr_expected(arg, pt),
                        None => self.check_expr(arg),
                    };
                    self.check_and_record_value_use(arg, &arg_type);
                    if let Some(pt) = param_type {
                        self.recover(self.types_compatible(pt, &arg_type), ());
                    }
                }
                continue;
            }

            if let (
                HirParam::Normal {
                    multi_place: None, ..
                },
                HirExpr::Ref {
                    expr,
                    ref_kind: rk @ (RefKind::Unique | RefKind::Alias),
                    span,
                },
            ) = (param, arg)
            {
                let ignores_init = Self::callee_ignores_initial_contents(
                    write_template.as_deref(),
                    read_template.as_deref(),
                    i,
                );
                let arg_type =
                    self.check_ref_expr_deferring_init(expr, *rk, *span, true, ignores_init);
                if let Some(pt) = param_type {
                    self.recover(self.types_compatible(pt, &arg_type), ());
                }
                if ignores_init {
                    self.optimistically_mark_mut_target_init(expr);
                }
                if let Some(place) = self.resolve_place(expr) {
                    if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                        arg_loans.push(loan_id);
                    }
                }
                continue;
            }

            let arg_type = match param_type {
                Some(pt) => self.check_expr_expected(arg, pt),
                None => self.check_expr(arg),
            };
            self.check_and_record_value_use(arg, &arg_type);
            if let Some(pt) = param_type {
                self.recover(self.types_compatible(pt, &arg_type), ());
            }

            if let HirExpr::Ref { expr, .. } = arg {
                if let Some(place) = self.resolve_place(expr) {
                    if let Some(&loan_id) = self.borrow_checker.loan_for_place(place) {
                        arg_loans.push(loan_id);
                    }
                }
            }
        }

        arg_loans
    }

    fn return_type_may_alias(&self, ty: &HirType<'a, 'bump>) -> bool {
        match ty {
            HirType::Ref { .. }
            | HirType::SafePointer { .. }
            | HirType::UnsafePointer { .. }
            | HirType::OwnedPointer { .. } => true,

            HirType::Nullable(inner) => self.return_type_may_alias(inner),

            HirType::Array(inner, _) => self.return_type_may_alias(inner),

            HirType::Tuple(elems) => elems.iter().any(|e| self.return_type_may_alias(e)),

            HirType::Struct { name, .. } => {
                let name_str = self.str_id_to_string(*name);
                match self.context.get_struct(&name_str) {
                    Some(def) => def
                        .fields
                        .iter()
                        .any(|f| self.return_type_may_alias(&f.field_type)),
                    None => true,
                }
            }

            HirType::Enum { name, .. } => {
                let name_str = self.str_id_to_string(*name);
                match self.context.get_enum(&name_str) {
                    Some(def) => def
                        .variants
                        .iter()
                        .flat_map(|v| v.fields.iter())
                        .any(|f| self.return_type_may_alias(&f.field_type)),
                    None => true,
                }
            }

            HirType::Dyn { .. } | HirType::DynInterface(..) => true,

            _ => false,
        }
    }

    fn path_display(path: &[StrId], range: Option<&UsedIndex>) -> String {
        let mut s = String::new();
        for seg in path {
            s.push('.');
            s.push_str(&seg.to_string());
        }
        if let Some(idx) = range {
            match idx {
                UsedIndex::Range(start, end) if *end == *start + 1 => {
                    s.push_str(&format!("[{}]", start));
                }
                UsedIndex::Range(start, end) => {
                    s.push_str(&format!("[{}..{}]", start, end));
                }
                UsedIndex::Place(root, p) => {
                    s.push('[');
                    s.push_str(&root.to_string());
                    for seg in p {
                        s.push('.');
                        s.push_str(&seg.to_string());
                    }
                    s.push(']');
                }
            }
        }
        s
    }

    fn finalize_call_loans(
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

    fn analyze_ref_template(&mut self, func: &HirFunc<'a, 'bump>) -> RefTemplate {
        if let Some(t) = self.ref_templates.get(&func.name) {
            return t.clone();
        }

        self.ref_templates.insert(func.name, RefTemplate::Opaque);

        let template = Self::build_ref_template(func);
        self.ref_templates.insert(func.name, template.clone());
        template
    }

    fn check_return_provenance(&mut self, func: &HirFunc<'a, 'bump>) {
        let Some(HirType::Ref {
            provenance: Some(ann),
            ..
        }) = func.return_type
        else {
            return;
        };

        let template = Self::build_ref_template(func);
        let RefTemplate::Path { base, .. } = template else {
            self.record(TypeErrorKind::Generic(format!(
                "return type declares provenance `{}` but the body's returned reference isn't a simple projection",
                self.provenance_to_string(&ann)
            )));
            return;
        };

        let root_matches = match (ann.root, base) {
            (ProvenanceRoot::Var(name), TemplateBase::Param(idx)) => func
                .params
                .map(|p| {
                    p.iter()
                        .filter(|pp| matches!(pp, HirParam::Normal { .. }))
                        .collect::<Vec<_>>()
                })
                .and_then(|normals| normals.get(idx).copied())
                .is_some_and(
                    |p| matches!(p, HirParam::Normal { name: pname, .. } if name == *pname),
                ),
            (ProvenanceRoot::ThisRoot, TemplateBase::This) => true,
            _ => false,
        };

        if !root_matches {
            self.record(TypeErrorKind::Generic(format!(
                "declared provenance `{}` doesn't match the parameter the returned reference is actually rooted in",
                self.provenance_to_string(&ann)
            )));
        }
    }

    fn build_ref_template(func: &HirFunc<'a, 'bump>) -> RefTemplate {
        let Some(params) = func.params else {
            return RefTemplate::Opaque;
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

        let Some(HirStmt::Block { body, span: _ }) = func.body else {
            return RefTemplate::Opaque;
        };

        let [HirStmt::Return(Some(ret_expr), _span)] = body else {
            return RefTemplate::Opaque;
        };
        let (expr, ref_kind) = match ret_expr {
            HirExpr::Ref {
                expr,
                ref_kind: mutable,
                ..
            } => (*expr, *mutable),
            HirExpr::Intrinsic {
                kind: IntrinsicKind::Own,
                args,
                ..
            } if args.len() == 2 => (&args[0], RefKind::Unique),
            _ => return RefTemplate::Opaque,
        };

        let Some((base, projections)) = Self::expr_to_template(expr, &param_index, has_this) else {
            return RefTemplate::Opaque;
        };

        RefTemplate::Path {
            base,
            ref_kind,
            projections,
        }
    }

    fn expr_to_template(
        expr: &HirExpr<'a, 'bump>,
        param_index: &FxHashMap<StrId, usize>,
        has_this: bool,
    ) -> Option<(TemplateBase, Vec<TemplateProjection>)> {
        match expr {
            HirExpr::Ident(name, _) => {
                Some((TemplateBase::Param(*param_index.get(name)?), Vec::new()))
            }

            HirExpr::This { .. } if has_this => Some((TemplateBase::This, Vec::new())),

            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                let (base, mut proj) = Self::expr_to_template(object, param_index, has_this)?;
                proj.push(TemplateProjection::Field(*field));
                Some((base, proj))
            }

            HirExpr::Deref { expr, .. } => {
                let (base, mut proj) = Self::expr_to_template(expr, param_index, has_this)?;
                proj.push(TemplateProjection::Deref);
                Some((base, proj))
            }

            HirExpr::Index { object, index, .. } => {
                let (base, mut proj) = Self::expr_to_template(object, param_index, has_this)?;
                let idx = match &**index {
                    HirExpr::Number(n, _) => IndexTemplate::Const(*n),
                    HirExpr::Ident(name, _) => param_index
                        .get(name)
                        .map(|&i| IndexTemplate::Param(i))
                        .unwrap_or(IndexTemplate::Opaque),
                    _ => IndexTemplate::Opaque,
                };
                proj.push(TemplateProjection::Index(idx));
                Some((base, proj))
            }

            _ => None,
        }
    }

    fn provenance_from_template(
        &self,
        template: &RefTemplate,
        receiver: Option<&HirExpr<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
    ) -> Option<ProvenanceAnnotation<'bump>> {
        let RefTemplate::Path {
            base, projections, ..
        } = template
        else {
            return None;
        };

        let base_expr = match base {
            TemplateBase::This => receiver?,
            TemplateBase::Param(i) => {
                let arg = args.get(*i)?;
                match arg {
                    HirExpr::Ref { expr, .. } => expr,
                    other => other,
                }
            }
        };

        let mut base_provenance = self.infer_provenance(base_expr)?;

        let mut path: Vec<ProvenancePathSegment> = base_provenance.path.to_vec();
        for proj in projections {
            match proj {
                TemplateProjection::Field(f) => path.push(ProvenancePathSegment::Field(*f)),
                TemplateProjection::Deref => path.push(ProvenancePathSegment::Deref),
                TemplateProjection::Index(_) => {}
            }
        }
        base_provenance.path = self.context.bump.alloc_slice(&path);
        Some(base_provenance)
    }

    fn resolve_template_place(
        &mut self,
        template: &RefTemplate,
        receiver: Option<&HirExpr<'a, 'bump>>,
        args: &[HirExpr<'a, 'bump>],
    ) -> Option<PlaceId> {
        let RefTemplate::Path {
            base, projections, ..
        } = template
        else {
            return None;
        };

        let base_expr = match base {
            TemplateBase::This => receiver?,
            TemplateBase::Param(i) => {
                let arg = args.get(*i)?;
                match arg {
                    HirExpr::Ref { expr, .. } => expr,
                    other => other,
                }
            }
        };

        let mut place = self.resolve_place(base_expr)?;

        for proj in projections {
            place = match proj {
                TemplateProjection::Field(f) => self.borrow_checker.project_field(place, *f),
                TemplateProjection::Deref => self.borrow_checker.project_deref(place),
                TemplateProjection::Index(idx_template) => {
                    let bound = match idx_template {
                        IndexTemplate::Const(c) => Bound::Const(*c),
                        IndexTemplate::Param(i) => self.expr_to_bound(args.get(*i)?),
                        IndexTemplate::Opaque => Bound::Opaque(0),
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

        Some(place)
    }

    fn check_receiver_is_mutable(
        &self,
        receiver: &HirExpr<'a, 'bump>,
        method_name: &str,
    ) -> TypeCheckResult<'a, ()> {
        let Some(root_name) = self.find_root_local_ident(receiver) else {
            return Ok(());
        };

        if !self.context.is_mutable(&root_name) {
            return Err(TypeErrorKind::Generic(format!(
                "cannot call `{}` on `{}`: `{}` is not declared `mut`",
                method_name, root_name, root_name
            ))
            .at(self.current_span));
        }

        Ok(())
    }

    fn expr_is_dangling(&self, expr: &HirExpr<'a, 'bump>) -> bool {
        match expr {
            HirExpr::Ref { expr: inner, .. } => match self.find_root_local_ident(inner) {
                Some(root_name) => match self.context.get_variable(&root_name) {
                    Some((_, root_type)) => !root_type.is_pointer_semantics(),
                    None => false,
                },
                None => false,
            },
            HirExpr::Ident(name, _) => {
                let var_name = self.str_id_to_string(*name);
                self.context.is_dangling(&var_name)
            }
            _ => false,
        }
    }

    fn check_no_dangling_pointer(&self, expr: &HirExpr<'a, 'bump>) -> TypeCheckResult<'a, ()> {
        if let HirExpr::Ref { expr: inner, .. } = expr {
            if let Some(root_name) = self.find_root_local_ident(inner) {
                if let Some((_, root_type)) = self.context.get_variable(&root_name) {
                    if !root_type.is_pointer_semantics() {
                        return Err(TypeErrorKind::Generic(format!(
                            "cannot return a pointer to local variable `{}`: its storage does not outlive this function",
                            root_name
                        )).at(self.current_span));
                    }
                }
            }
            return Ok(());
        }

        if let HirExpr::Ident(name, _) = expr {
            let var_name = self.str_id_to_string(*name);
            if self.context.is_dangling(&var_name) {
                return Err(TypeErrorKind::Generic(format!(
                    "cannot return `{}`: it holds a pointer to local stack memory that does not outlive this function",
                    var_name
                )).at(self.current_span));
            }
        }

        Ok(())
    }

    fn find_root_local_ident(&self, expr: &HirExpr<'a, 'bump>) -> Option<String> {
        match expr {
            HirExpr::Ident(name, _) => Some(self.str_id_to_string(*name)),
            HirExpr::FieldAccess { object, .. } | HirExpr::Get { object, .. } => {
                self.find_root_local_ident(object)
            }
            HirExpr::Deref { expr: inner, .. } => self.find_root_local_ident(inner),
            _ => None,
        }
    }

    fn check_binary_op(
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
                        op: self.operator_symbol(op),
                        left: self.type_to_string(left),
                        right: self.type_to_string(right),
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
                        op: self.operator_symbol(op),
                        left: self.type_to_string(left),
                        right: self.type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            LessThan | LessThanOrEqual | GreaterThan | GreaterThanOrEqual => {
                if self.is_comparable(left) && self.is_comparable(right) {
                    Ok(HirType::Boolean)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: self.operator_symbol(op),
                        left: self.type_to_string(left),
                        right: self.type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            LogicalAnd | LogicalOr => {
                if *left == HirType::Boolean && *right == HirType::Boolean {
                    Ok(HirType::Boolean)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: self.operator_symbol(op),
                        left: self.type_to_string(left),
                        right: self.type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            BitAnd | BitOr | BitXor | ShiftLeft | ShiftRight => {
                if self.is_integer(left) && self.is_integer(right) {
                    Ok(*left)
                } else {
                    Err(TypeErrorKind::InvalidBinaryOp {
                        op: self.operator_symbol(op),
                        left: self.type_to_string(left),
                        right: self.type_to_string(right),
                    }
                    .at(self.current_span))
                }
            }

            _ => Err(TypeErrorKind::Generic(format!(
                "operator `{}` cannot appear in this position",
                self.operator_symbol(op)
            ))
            .at(self.current_span)),
        }
    }

    fn operator_symbol(&self, op: &Operator) -> String {
        use Operator::*;
        match op {
            Add => "+",
            Subtract => "-",
            Multiply => "*",
            Divide => "/",
            Modulo => "%",
            Equals => "==",
            NotEquals => "!=",
            LessThan => "<",
            LessThanOrEqual => "<=",
            GreaterThan => ">",
            GreaterThanOrEqual => ">=",
            LogicalAnd => "&&",
            LogicalOr => "||",
            BitAnd => "&",
            BitOr => "|",
            BitXor => "^",
            ShiftLeft => "<<",
            ShiftRight => ">>",
            _ => return format!("{:?}", op), // TODO finish
        }
        .to_string()
    }

    fn is_reference_like(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::Ref { .. }
                | HirType::SafePointer { .. }
                | HirType::UnsafePointer { .. }
                | HirType::OwnedPointer { .. }
        )
    }

    fn nullable_equality_compatible(
        &self,
        left: &HirType<'a, 'bump>,
        right: &HirType<'a, 'bump>,
    ) -> bool {
        let unwrap_nullable = |t: &HirType<'a, 'bump>| match t {
            HirType::Nullable(inner) => Some(**inner),
            _ => None,
        };

        let payload_compatible = |a: &HirType<'a, 'bump>, b: &HirType<'a, 'bump>| {
            (self.is_comparable(a) && self.is_comparable(b) && self.types_structurally_equal(a, b))
                || (self.is_reference_like(a)
                    && self.is_reference_like(b)
                    && self.types_structurally_equal(a, b))
        };

        match (unwrap_nullable(left), unwrap_nullable(right)) {
            (Some(li), Some(ri)) => payload_compatible(&li, &ri),
            (Some(li), None) => payload_compatible(&li, right) || matches!(right, HirType::Null),
            (None, Some(ri)) => payload_compatible(left, &ri) || matches!(left, HirType::Null),
            (None, None) => false,
        }
    }

    fn instantiate_struct(
        &self,
        name: StrId,
        args: &[HirType<'a, 'bump>],
    ) -> Option<&'bump [HirType<'a, 'bump>]> {
        if let Some(cached) = self.context.get_struct_instantiation(name, args) {
            return Some(cached);
        }

        let name_str = self.str_id_to_string(name);
        let def = self.context.get_struct(&name_str)?;
        let generics = def.generics?;
        let mut full_args = args.to_vec();
        if full_args.len() < generics.len() {
            let mut subs = FxHashMap::default();
            for (param, arg) in generics.iter().zip(full_args.iter()) {
                subs.insert(param.name, *arg);
            }
            let start = full_args.len();
            for param in &generics[start..] {
                if let Some(ref def_ty) = param.default_type {
                    let resolved = self.substitute_type_local(def_ty, &subs);
                    subs.insert(param.name, resolved);
                    full_args.push(resolved);
                } else {
                    return None;
                }
            }
        }
        if generics.len() != full_args.len() {
            return None;
        }

        let mut subs = FxHashMap::default();
        for (param, arg) in generics.iter().zip(full_args.iter()) {
            subs.insert(param.name, *arg);
        }

        let field_types: Vec<_> = def
            .fields
            .iter()
            .map(|f| self.substitute_type_local(&f.field_type, &subs))
            .collect();
        let result = self.context.bump.alloc_slice_copy(&field_types);

        self.context
            .cache_struct_instantiation(name, args.to_vec(), result);
        Some(result)
    }

    fn check_field_access(
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
            if self.str_id_to_string(field) == "len" {
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
                self.type_to_string(&obj_type)
            )));
            return HirType::Unknown;
        };

        let struct_name_str = self.str_id_to_string(struct_name);
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

        let field_name = self.str_id_to_string(field);
        let field_idx = struct_def
            .fields
            .iter()
            .position(|f| self.str_id_to_string(f.name) == field_name);

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
            let root_str = self.str_id_to_string(root);
            self.check_init_read_path(root, &path, &root_str);
        }

        ty
    }

    fn record_move(
        &mut self,
        root: StrId,
        field: Option<StrId>,
        field_ty: &HirType<'a, 'bump>,
        container_ty: Option<&HirType<'a, 'bump>>,
    ) {
        if self.copy_analysis.borrow().type_is_copy(field_ty) {
            return;
        }

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
                        container_ty
                            .map(|t| self.type_to_string(t))
                            .unwrap_or_default()
                    )));
                    return;
                }

                self.move_state.mark_field_moved(root, f);
            }
        }
    }

    fn check_use(&mut self, root: StrId, field: Option<StrId>, root_ty: &HirType<'a, 'bump>) {
        if self.copy_analysis.borrow().type_is_copy(root_ty) {
            return;
        }
        let name = self.str_id_to_string(root);
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
                        self.str_id_to_string(f)
                    )));
                }
            }
        }
    }

    fn check_and_record_value_use(&mut self, expr: &HirExpr<'a, 'bump>, ty: &HirType<'a, 'bump>) {
        if let Some(place) = self.resolve_place(expr) {
            self.check_borrow_use(expr, place, BorrowKind::Shared);
        }
        match expr {
            HirExpr::Ident(name, _) => {
                self.check_use(*name, None, ty);
                self.record_move(*name, None, ty, None);
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                if let HirExpr::Ident(root, _) = &**object {
                    let root_name = self.str_id_to_string(*root);
                    let container_ty = self.context.get_variable(&root_name);
                    self.check_use(*root, Some(*field), ty);
                    self.record_move(*root, Some(*field), ty, container_ty.map(|f| f.1).as_ref());
                }
            }
            _ => {}
        }
    }

    fn types_structurally_equal(&self, a: &HirType<'a, 'bump>, b: &HirType<'a, 'bump>) -> bool {
        use HirType::*;
        if matches!(a, Generic(_)) || matches!(b, Generic(_)) {
            return true;
        }
        match (a, b) {
            (
                Struct {
                    name: na,
                    type_args: ta,
                    ..
                },
                Struct {
                    name: nb,
                    type_args: tb,
                    ..
                },
            ) => {
                na == nb
                    && ta.len() == tb.len()
                    && ta
                        .iter()
                        .zip(tb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            (
                Ref {
                    inner: ia,
                    ref_kind: ma,
                    ..
                },
                Ref {
                    inner: ib,
                    ref_kind: mb,
                    ..
                },
            ) => {
                // Deliberately ignore provenance here, it's borrow-checker
                // bookkeeping about where a reference came from, not part of
                // the reference's type identity. Two `&mut i64` are the same
                // type regardless of which place each one happens to alias.
                ma == mb && self.types_structurally_equal(ia, ib)
            }
            (Nullable(ia), Nullable(ib)) => self.types_structurally_equal(ia, ib),
            (
                SafePointer {
                    inner: ia,
                    mutability_state: ma,
                },
                SafePointer {
                    inner: ib,
                    mutability_state: mb,
                },
            ) => ma == mb && self.types_structurally_equal(ia, ib),
            (
                UnsafePointer {
                    inner: ia,
                    mutability_state: ma,
                },
                UnsafePointer {
                    inner: ib,
                    mutability_state: mb,
                },
            ) => ma == mb && self.types_structurally_equal(ia, ib),
            (
                OwnedPointer {
                    inner: ia,
                    allocator: aa,
                },
                OwnedPointer {
                    inner: ib,
                    allocator: ab,
                },
            ) => aa == ab && self.types_structurally_equal(ia, ib),
            (Array(ia, la), Array(ib, lb)) => la == lb && self.types_structurally_equal(ia, ib),
            (Slice(ia), Slice(ib)) => self.types_structurally_equal(ia, ib),
            (Tuple(ta), Tuple(tb)) => {
                ta.len() == tb.len()
                    && ta
                        .iter()
                        .zip(tb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            (Dyn { bounds: ba }, Dyn { bounds: bb }) => {
                ba.len() == bb.len()
                    && ba
                        .iter()
                        .zip(bb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            (
                Lambda {
                    params: pa,
                    return_type: ra,
                },
                Lambda {
                    params: pb,
                    return_type: rb,
                },
            ) => {
                pa.len() == pb.len()
                    && pa
                        .iter()
                        .zip(pb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
                    && self.types_structurally_equal(ra, rb)
            }
            (
                Enum {
                    name: na,
                    type_args: ta,
                    ..
                },
                Enum {
                    name: nb,
                    type_args: tb,
                    ..
                },
            ) => {
                na == nb
                    && ta.len() == tb.len()
                    && ta
                        .iter()
                        .zip(tb.iter())
                        .all(|(x, y)| self.types_structurally_equal(x, y))
            }
            _ => a == b,
        }
    }

    fn substitute_type_local(
        &self,
        ty: &HirType<'a, 'bump>,
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        match ty {
            HirType::Generic(name) => subs.get(name).copied().unwrap_or(*ty),
            HirType::Nullable(inner) => HirType::Nullable(
                self.context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
            ),
            HirType::Array(inner, len) => HirType::Array(
                self.context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                *len,
            ),
            HirType::Slice(inner) => HirType::Slice(
                self.context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
            ),
            HirType::SafePointer {
                inner,
                mutability_state,
            } => HirType::SafePointer {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                mutability_state: *mutability_state,
            },
            HirType::UnsafePointer {
                inner,
                mutability_state,
            } => HirType::UnsafePointer {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                mutability_state: *mutability_state,
            },
            HirType::OwnedPointer { inner, allocator } => HirType::OwnedPointer {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                allocator: *allocator,
            },
            HirType::Ref {
                inner,
                ref_kind: mutability_state,
                provenance,
            } => HirType::Ref {
                inner: self
                    .context
                    .bump
                    .alloc_value(self.substitute_type_local(inner, subs)),
                ref_kind: *mutability_state,
                provenance: *provenance,
            },
            HirType::Tuple(elems) => {
                let new_elems: Vec<_> = elems
                    .iter()
                    .map(|e| self.substitute_type_local(e, subs))
                    .collect();
                HirType::Tuple(self.context.bump.alloc_slice_copy(&new_elems))
            }

            HirType::Struct {
                name,
                field_types,
                type_args,
            } => {
                let new_fields: Vec<_> = field_types
                    .iter()
                    .map(|f| self.substitute_type_local(f, subs))
                    .collect();
                let new_args: Vec<_> = type_args
                    .iter()
                    .map(|a| self.substitute_type_local(a, subs))
                    .collect();
                HirType::Struct {
                    name: *name,
                    field_types: self.context.bump.alloc_slice_copy(&new_fields),
                    type_args: self.context.bump.alloc_slice_copy(&new_args),
                }
            }
            HirType::Enum {
                name,
                type_args,
                variants,
            } => {
                let new_args: Vec<_> = type_args
                    .iter()
                    .map(|a| self.substitute_type_local(a, subs))
                    .collect();
                HirType::Enum {
                    name: *name,
                    type_args: self.context.bump.alloc_slice_copy(&new_args),
                    variants,
                }
            }
            HirType::Dyn { bounds } => {
                let new_bounds: Vec<_> = bounds
                    .iter()
                    .map(|b| self.substitute_type_local(b, subs))
                    .collect();
                HirType::Dyn {
                    bounds: self.context.bump.alloc_slice_copy(&new_bounds),
                }
            }
            HirType::Lambda {
                params,
                return_type,
            } => {
                let new_params: Vec<_> = params
                    .iter()
                    .map(|p| self.substitute_type_local(p, subs))
                    .collect();
                HirType::Lambda {
                    params: self.context.bump.alloc_slice_copy(&new_params),
                    return_type: self
                        .context
                        .bump
                        .alloc_value(self.substitute_type_local(return_type, subs)),
                }
            }

            _ => *ty,
        }
    }

    ///   Public:   visible everywhere.
    ///   Module:   visible only within the declaring module (DOESN'T WORK NOW)
    ///   Private:  visibile only within the same file
    ///   Internal: visible anywhere in the same package, not outside it.
    fn check_visibility(
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

    fn substitute_params_local(
        &self,
        params: &[HirParam<'a, 'bump>],
        subs: &FxHashMap<StrId, HirType<'a, 'bump>>,
    ) -> &'bump [HirParam<'a, 'bump>] {
        let new_params: Vec<HirParam> = params
            .iter()
            .map(|p| match p {
                HirParam::Normal {
                    name,
                    param_type,
                    multi_place,
                    span,
                } => HirParam::Normal {
                    name: *name,
                    param_type: self.substitute_type_local(param_type, subs),
                    multi_place: *multi_place,
                    span: *span,
                },
                HirParam::This {
                    kind,
                    span,
                    multi_place,
                } => HirParam::This {
                    kind: *kind,
                    span: *span,
                    multi_place: *multi_place,
                },
            })
            .collect();
        self.context.bump.alloc_slice(&new_params)
    }

    fn types_compatible(
        &self,
        expected: &HirType<'a, 'bump>,
        found: &HirType<'a, 'bump>,
    ) -> TypeCheckResult<'a, ()> {
        if matches!(found, HirType::Never) {
            return Ok(());
        }

        if let (
            HirType::Ref {
                inner: ei,
                ref_kind: erk,
                ..
            },
            HirType::Ref {
                inner: fi,
                ref_kind: frk,
                ..
            },
        ) = (expected, found)
        {
            if self.types_structurally_equal(ei, fi) && frk.coerces_to(*erk) {
                return Ok(());
            }
        }

        if self.types_structurally_equal(expected, found) {
            return Ok(());
        }

        if *expected == HirType::Unknown || *found == HirType::Unknown {
            return Ok(());
        }

        if self.struct_satisfies_interface_type(expected, found)
            || self.struct_satisfies_interface_type(found, expected)
        {
            return Ok(());
        }

        if let HirType::Nullable(_) = expected {
            if found == &HirType::Null {
                return Ok(());
            }
        }

        Err(TypeErrorKind::TypeMismatch {
            expected: self.type_to_string(expected),
            found: self.type_to_string(found),
        }
        .at(self.current_span))
    }

    fn struct_satisfies_interface_type(
        &self,
        expected: &HirType<'a, 'bump>,
        found: &HirType<'a, 'bump>,
    ) -> bool {
        let expected_inner = Self::strip_ref(expected);
        let found_inner = Self::strip_ref(found);

        let interface_name = match expected_inner {
            HirType::DynInterface(name, _) => Some(name.to_string()),
            HirType::Dyn { bounds } => bounds.iter().find_map(|b| match b {
                HirType::DynInterface(name, _) => Some(name.to_string()),
                HirType::Struct { name, .. } => {
                    let name_str = name.to_string();
                    if self.context.get_interface(&name_str).is_some() {
                        Some(name_str)
                    } else {
                        None
                    }
                }
                _ => None,
            }),
            _ => None,
        };

        let Some(interface_name) = interface_name else {
            return false;
        };

        let struct_name = match found_inner {
            HirType::Struct { name, .. } => name.to_string(),
            _ => return false,
        };

        self.context
            .struct_implements(&struct_name, &interface_name)
    }

    fn check_slice_primitive_call(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        elem: HirType<'a, 'bump>,
        method: &str,
        args: &[HirExpr<'a, 'bump>],
    ) -> HirType<'a, 'bump> {
        let ret = if method == "get_unchecked" {
            elem
        } else {
            HirType::Void
        };

        if !self.in_unsafe() {
            self.record(TypeErrorKind::Generic(format!(
                "`{}` requires an unsafe block: it performs no bounds check",
                method
            )));
        }

        let (expected_args, needs_mut) = match method {
            "write_uninit" => (2, true),
            "write_uninit_all" => (1, true),
            _ => (1, false),
        };
        if args.len() != expected_args {
            self.record(TypeErrorKind::InvalidFunctionCall {
                expected_args,
                found_args: args.len(),
            });
            for a in args {
                self.check_expr(a);
            }
            return ret;
        }

        if needs_mut {
            self.recover(self.check_receiver_is_mutable(object, method), ());
        }
        if let Some(place) = self.resolve_place(object) {
            let kind = if needs_mut {
                BorrowKind::Mutable
            } else {
                BorrowKind::Shared
            };
            self.check_borrow_use(object, place, kind);
        }

        let idx_or_src_ty = |this: &mut Self, e: &HirExpr<'a, 'bump>| {
            let t = this.check_expr_expected(e, &HirType::Usize);
            if !this.is_integer(&t) {
                this.record(TypeErrorKind::Generic(format!(
                    "index must be an integer, found `{}`",
                    this.type_to_string(&t)
                )));
            }
        };

        match method {
            "write_uninit" => {
                idx_or_src_ty(self, &args[0]);
                let val_ty = self.check_expr_expected(&args[1], &elem);
                self.check_and_record_value_use(&args[1], &val_ty);
                self.recover(self.types_compatible(&elem, &val_ty), ());
            }
            "get_unchecked" => idx_or_src_ty(self, &args[0]),
            "write_uninit_all" => {
                let src_ty = self.check_expr(&args[0]);
                match *Self::strip_ref(&src_ty) {
                    HirType::Slice(e) | HirType::Array(e, _) => {
                        self.recover(self.types_compatible(&elem, e), ());
                    }
                    _ => self.record(TypeErrorKind::Generic(format!(
                        "`write_uninit_all` expects a slice or array source, found `{}`",
                        self.type_to_string(&src_ty)
                    ))),
                }
                if let Some(p) = self.resolve_place(&args[0]) {
                    self.check_borrow_use(&args[0], p, BorrowKind::Shared);
                }
            }
            _ => unreachable!(),
        }
        ret
    }

    fn strip_ref<'x>(ty: &'x HirType<'a, 'bump>) -> &'x HirType<'a, 'bump> {
        match ty {
            HirType::Ref { inner, .. } => inner,
            HirType::SafePointer { inner, .. } => inner,
            HirType::OwnedPointer { inner, .. } => inner,
            _ => ty,
        }
    }

    fn is_numeric(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::F32
                | HirType::F64
                | HirType::I128
                | HirType::U128
                | HirType::Usize
                | HirType::Isize
        )
    }

    fn is_integer(&self, ty: &HirType<'a, 'bump>) -> bool {
        matches!(
            ty,
            HirType::I8
                | HirType::I16
                | HirType::I32
                | HirType::I64
                | HirType::U8
                | HirType::U16
                | HirType::U32
                | HirType::U64
                | HirType::I128
                | HirType::U128
                | HirType::Usize
                | HirType::Isize
        )
    }

    fn is_comparable(&self, ty: &HirType<'a, 'bump>) -> bool {
        self.is_numeric(ty) || matches!(ty, HirType::Boolean | HirType::String)
    }

    fn str_id_to_string(&self, id: StrId) -> String {
        format!("{}", id)
    }

    fn type_to_string(&self, ty: &HirType<'a, 'bump>) -> String {
        match ty {
            HirType::I8 => "i8".to_string(),
            HirType::I16 => "i16".to_string(),
            HirType::I32 => "i32".to_string(),
            HirType::I64 => "i64".to_string(),
            HirType::U8 => "u8".to_string(),
            HirType::U16 => "u16".to_string(),
            HirType::U32 => "u32".to_string(),
            HirType::U64 => "u64".to_string(),
            HirType::F32 => "f32".to_string(),
            HirType::F64 => "f64".to_string(),
            HirType::I128 => "i128".to_string(),
            HirType::U128 => "u128".to_string(),
            HirType::Boolean => "bool".to_string(),
            HirType::String => "str".to_string(),
            HirType::Void => "void".to_string(),
            HirType::Unknown => "<unknown>".to_string(),
            HirType::Struct {
                name, type_args, ..
            } => {
                if type_args.is_empty() {
                    format!("struct {}", self.str_id_to_string(*name))
                } else {
                    let args = type_args
                        .iter()
                        .map(|t| self.type_to_string(t))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("struct {}<{}>", self.str_id_to_string(*name), args)
                }
            }
            HirType::DynInterface(name, _) => format!("interface {}", self.str_id_to_string(*name)),
            HirType::Enum {
                name, type_args, ..
            } => {
                if type_args.is_empty() {
                    format!("enum {}", self.str_id_to_string(*name))
                } else {
                    let args = type_args
                        .iter()
                        .map(|t| self.type_to_string(t))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("enum {}<{}>", self.str_id_to_string(*name), args)
                }
            }
            HirType::Generic(name) => format!("generic {}", self.str_id_to_string(*name)),
            HirType::SafePointer {
                inner,
                mutability_state,
            } => format!("*{} {}", mutability_state, inner),
            HirType::UnsafePointer {
                inner,
                mutability_state,
            } => format!("[*]{} {}", mutability_state, inner),
            HirType::Lambda { .. } => "lambda".to_string(),
            HirType::This => "this".to_string(),
            HirType::Null => "null".to_string(),
            HirType::Char => "char".to_string(),
            HirType::Ref {
                inner,
                ref_kind,
                provenance,
            } => {
                let displayed_provenance = if let Some(provenance) = provenance {
                    format!("{}", provenance)
                } else {
                    String::new()
                };
                if let RefKind::Unique = *ref_kind {
                    format!("&{}mut {}", displayed_provenance, inner)
                } else if let RefKind::Alias = *ref_kind {
                    format!("&{}alias {}", displayed_provenance, inner)
                } else {
                    format!("&{}{}", displayed_provenance, inner)
                }
            }
            HirType::Nullable(hir_type) => format!("?{}", hir_type),
            HirType::Dyn { bounds } => {
                let mut bounds_str = String::new();
                let mut start = true;
                for bound in *bounds {
                    bounds_str.push_str(&bound.to_string());
                    if start {
                        start = false;
                    } else {
                        bounds_str.push_str(" + ");
                    }
                }
                format!("dyn {}", bounds_str)
            }
            HirType::Tuple(hir_types) => format!(
                "({})",
                hir_types
                    .iter()
                    .map(|t| self.type_to_string(t))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            HirType::Array(inner, len) => format!("[{}]{}", len, self.type_to_string(inner)),
            HirType::Slice(inner) => format!("[]{}", self.type_to_string(inner)),
            HirType::OwnedPointer { inner, allocator } => {
                format!(
                    "^{} {}",
                    allocator
                        .map(|all| self.provenance_to_string(&all))
                        .unwrap_or(String::from("")),
                    self.type_to_string(inner)
                )
            }
            HirType::Usize => "usize".to_string(),
            HirType::Isize => "isize".to_string(),
            HirType::Never => "never".to_string(),
            HirType::Range { elem, inclusive } => format!(
                "range<{}>{}",
                self.type_to_string(elem),
                if *inclusive { " (inclusive)" } else { "" }
            ),
        }
    }

    fn describe_borrow_error(
        &self,
        err: &BorrowError,
        provenance: Option<&ProvenanceAnnotation>,
    ) -> String {
        let base = match err {
            BorrowError::UseAfterMove { .. } => "use of a value after it was moved".to_string(),
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
            Some(p) => format!("{} (via {})", base, self.provenance_to_string(p)),
            None => base.to_string(),
        }
    }

    fn provenance_to_string(&self, p: &ProvenanceAnnotation) -> String {
        let root = match p.root {
            ProvenanceRoot::Var(name) => self.str_id_to_string(name),
            ProvenanceRoot::ThisRoot => "this".to_string(),
            ProvenanceRoot::Global {
                module_idx: _,
                name,
            } => self.str_id_to_string(name),
            ProvenanceRoot::ImplicitParam(_) => todo!(),
        };
        p.path.iter().fold(root, |acc, seg| match seg {
            ProvenancePathSegment::Field(f) => format!("{}.{}", acc, self.str_id_to_string(*f)),
            ProvenancePathSegment::Deref => format!("*{}", acc),
        })
    }

    fn check_slice_range_init(
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
        let root_str = self.str_id_to_string(root);

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

    fn peek_type(&self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        match expr {
            HirExpr::Ident(name, _) => {
                let var_name = self.str_id_to_string(*name);
                self.context
                    .get_variable(&var_name)
                    .unwrap_or((SymbolId::Local(LocalSymbolId(u32::MAX)), HirType::Unknown))
                    .1
            }
            HirExpr::This { .. } => {
                self.context
                    .get_variable("this")
                    .unwrap_or((SymbolId::Local(LocalSymbolId(u32::MAX)), HirType::This))
                    .1
            }
            HirExpr::FieldAccess { object, field, .. } | HirExpr::Get { object, field, .. } => {
                match self.peek_type(object) {
                    HirType::Struct {
                        name: struct_name,
                        type_args,
                        ..
                    } => {
                        let struct_name_str = self.str_id_to_string(struct_name);
                        let field_name = self.str_id_to_string(*field);
                        let Some(struct_def) = self.context.get_struct(&struct_name_str) else {
                            return HirType::Unknown;
                        };
                        let Some(field_idx) = struct_def
                            .fields
                            .iter()
                            .position(|f| self.str_id_to_string(f.name) == field_name)
                        else {
                            return HirType::Unknown;
                        };
                        if type_args.is_empty() {
                            struct_def.fields[field_idx].field_type
                        } else {
                            self.instantiate_struct(struct_name, type_args)
                                .map(|fields| fields[field_idx])
                                .unwrap_or(struct_def.fields[field_idx].field_type)
                        }
                    }
                    _ => HirType::Unknown,
                }
            }
            HirExpr::Deref { expr, .. } => match self.peek_type(expr) {
                HirType::Ref { inner, .. } => *inner,
                HirType::SafePointer { inner, .. } => *inner,
                HirType::UnsafePointer { inner, .. } => *inner,
                HirType::OwnedPointer { inner, .. } => *inner,
                _ => HirType::Unknown,
            },
            HirExpr::Index { object, .. } => match self.peek_type(object) {
                HirType::Array(inner, _) => *inner,
                HirType::Slice(inner) => *inner,
                _ => HirType::Unknown,
            },
            HirExpr::Cast { target_type, .. } => *target_type,
            _ => HirType::Unknown,
        }
    }

    fn resolve_place(&mut self, expr: &HirExpr<'a, 'bump>) -> Option<PlaceId> {
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

    fn fresh_opaque(&mut self) -> Bound {
        self.next_opaque_id += 1;
        Bound::Opaque(self.next_opaque_id)
    }

    fn expr_to_bound(&mut self, expr: &HirExpr<'a, 'bump>) -> Bound {
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

    fn loan_referent_place(&self, expr: &HirExpr<'a, 'bump>) -> Option<PlaceId> {
        let HirExpr::Ident(name, _) = expr else {
            return None;
        };
        let loan_id = self
            .loan_owners
            .iter()
            .find(|(_, &owner)| owner == *name)
            .map(|(&id, _)| id)?;
        self.borrow_checker.loan(loan_id).map(|loan| loan.place)
    }

    fn condition_to_place_fact(
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

    fn snapshot_call_loan_keys(&self) -> HashSet<usize> {
        self.call_loans.keys().copied().collect()
    }

    fn end_temp_call_loans(&mut self, before: &HashSet<usize>) {
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

    fn check_borrow_use(&mut self, expr: &HirExpr<'a, 'bump>, place: PlaceId, kind: BorrowKind) {
        if let Err(e) = self.borrow_checker.check_use(place, kind) {
            let provenance = self.infer_provenance(expr);
            let msg = self.describe_borrow_error(&e, provenance.as_ref());
            self.record(TypeErrorKind::Generic(msg));
        }
    }

    fn check_bare_name_import(
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

    fn generic_substitutions_for_struct(
        &self,
        struct_name: StrId,
        type_args: &[HirType<'a, 'bump>],
    ) -> FxHashMap<StrId, HirType<'a, 'bump>> {
        let mut subs = FxHashMap::default();
        if type_args.is_empty() {
            return subs;
        }
        let name_str = self.str_id_to_string(struct_name);
        if let Some(def) = self.context.get_struct(&name_str) {
            if let Some(generics) = def.generics {
                for (param, arg) in generics.iter().zip(type_args.iter()) {
                    subs.insert(param.name, *arg);
                }
            }
        }
        subs
    }

    fn check_match_exhaustiveness(
        &mut self,
        scrutinee_ty: &HirType<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
    ) {
        if matches!(scrutinee_ty, HirType::Unknown) {
            return;
        }

        let has_catch_all = arms.iter().any(|arm| {
            arm.guard.is_none()
                && matches!(arm.pattern, HirPattern::Wildcard | HirPattern::Ident(_))
        });
        if has_catch_all {
            return;
        }

        match scrutinee_ty {
            HirType::Nullable(_) => {
                let has_null = arms
                    .iter()
                    .any(|arm| arm.guard.is_none() && matches!(arm.pattern, HirPattern::Null));
                if !has_null {
                    self.record(TypeErrorKind::Generic(format!(
                        "non-exhaustive match on `{}`: missing a `null` arm",
                        self.type_to_string(scrutinee_ty)
                    )));
                }
            }
            HirType::Enum {
                name: enum_name, ..
            } => {
                let enum_name_str = self.str_id_to_string(*enum_name);
                let Some(def) = self.context.get_enum(&enum_name_str) else {
                    return;
                };
                let covered: std::collections::HashSet<StrId, FxHashBuilder> = arms
                    .iter()
                    .filter(|arm| arm.guard.is_none())
                    .filter_map(|arm| match &arm.pattern {
                        HirPattern::EnumVariant { variant, .. } => Some(*variant),
                        HirPattern::Struct { name, .. } => Some(*name),
                        _ => None,
                    })
                    .collect();
                let missing: Vec<&str> = def
                    .variants
                    .iter()
                    .filter(|v| !covered.contains(&v.name))
                    .map(|v| v.name.as_str())
                    .collect();
                if !missing.is_empty() {
                    self.record(TypeErrorKind::Generic(format!(
                        "non-exhaustive match on enum `{}`: missing variant(s) {}",
                        enum_name_str,
                        missing.join(", ")
                    )));
                }
            }

            HirType::Boolean => {
                let mut has_true = false;
                let mut has_false = false;
                for arm in arms.iter().filter(|a| a.guard.is_none()) {
                    match &arm.pattern {
                        HirPattern::Boolean(true) => has_true = true,
                        HirPattern::Boolean(false) => has_false = true,
                        _ => {}
                    }
                }
                if !(has_true && has_false) {
                    self.record(TypeErrorKind::Generic(
                        "non-exhaustive match on `bool`: requires a wildcard (`_`) arm or both `true` and `false` arms".to_string()
                    ));
                }
            }

            HirType::I8
            | HirType::I16
            | HirType::I32
            | HirType::I64
            | HirType::I128
            | HirType::U8
            | HirType::U16
            | HirType::U32
            | HirType::U64
            | HirType::U128
            | HirType::Usize
            | HirType::Isize
            | HirType::String
            | HirType::Char => {
                self.record(TypeErrorKind::Generic(format!(
                    "non-exhaustive match on `{}`: requires a wildcard (`_`) or binding (catch-all) arm",
                    self.type_to_string(scrutinee_ty)
                )));
            }

            _ => {} // structs/tuples/etc: not enforced yet
        }
    }

    fn local_used_after(&self, point: PointId, local: StrId) -> bool {
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

    fn unify_generic(
        &self,
        declared: &HirType<'a, 'bump>,
        actual: &HirType<'a, 'bump>,
        subs: &mut FxHashMap<StrId, HirType<'a, 'bump>>,
    ) {
        match declared {
            HirType::Generic(name) => {
                subs.entry(*name).or_insert(*actual);
            }
            HirType::Nullable(inner) => {
                if let HirType::Nullable(actual_inner) = actual {
                    self.unify_generic(inner, actual_inner, subs);
                }
            }
            HirType::Array(inner, _) => {
                if let HirType::Array(actual_inner, _) = actual {
                    self.unify_generic(inner, actual_inner, subs);
                }
            }
            HirType::Slice(inner) => {
                if let HirType::Slice(actual_inner) = actual {
                    self.unify_generic(inner, actual_inner, subs);
                }
            }
            HirType::Ref { inner, .. } => {
                self.unify_generic(inner, Self::strip_ref(actual), subs);
            }
            _ => {}
        }
    }

    fn scrutinee_binding_mode(&self, scrutinee: &HirExpr<'a, 'bump>) -> BindingMode {
        match scrutinee {
            HirExpr::Ref { ref_kind, .. } => BindingMode::ByRef(*ref_kind),
            HirExpr::This { .. } => match self.local_ref_kind.get(&self.this_id) {
                Some(rk) => BindingMode::ByRef(*rk),
                None => BindingMode::ByValue,
            },
            HirExpr::Ident(name, _) => match self.local_ref_kind.get(name) {
                Some(rk) => BindingMode::ByRef(*rk),
                None => BindingMode::ByValue,
            },
            _ => BindingMode::ByValue,
        }
    }

    pub fn binding_mode(&self, expr: &HirExpr<'a, 'bump>) -> BindingMode {
        self.binding_mode_backfill
            .get(&Self::expr_key(expr))
            .copied()
            .unwrap_or(BindingMode::ByValue)
    }

    fn check_enum_init(
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
        let enum_name_str = self.str_id_to_string(*enum_name);
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

        let variant_name = self.str_id_to_string(*variant);
        let variant_def = enum_def
            .variants
            .iter()
            .find(|v| self.str_id_to_string(v.name) == variant_name);
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

    fn check_tail_expected(
        &mut self,
        stmt: &HirStmt<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        match stmt {
            HirStmt::Expr(e) => match expected {
                Some(exp) => self.check_expr_expected(e, exp),
                None => self.check_expr(e),
            },
            HirStmt::Match { expr, arms, span } => {
                self.set_span(*span);
                self.check_match_arms(expr, arms, expected)
            }
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                self.set_span(*span);
                self.check_if_branches(cond, then_block, *else_block, expected)
                    .unwrap_or(HirType::Void)
            }
            HirStmt::Block { body, span } => {
                self.set_span(*span);
                self.check_block_body(body, expected)
                    .unwrap_or(HirType::Void)
            }
            other => self.check_stmt(other).unwrap_or(HirType::Void),
        }
    }

    fn check_block_tail_expected(
        &mut self,
        body: &HirStmt<'a, 'bump>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        let HirStmt::Block { body, span: _ } = body else {
            unreachable!()
        };
        let Some((last, rest)) = body.split_last() else {
            return HirType::Void;
        };
        for s in rest {
            self.check_stmt(s);
        }
        self.check_tail_expected(last, expected)
    }

    fn check_match_arms(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        arms: &[HirMatchArm<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> HirType<'a, 'bump> {
        let scrutinee_ty = self.check_expr(expr);

        self.check_match_exhaustiveness(&scrutinee_ty, arms);

        let move_state_before = self.move_state.clone();
        let mut arm_types = Vec::with_capacity(arms.len());

        let mut arm_move_states = Vec::with_capacity(arms.len());

        for arm in arms {
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
                        found: self.type_to_string(&guard_type),
                    });
                }
            }

            let arm_ty = self.check_block_tail_expected(arm.body, expected);
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

    fn check_block_body(
        &mut self,
        body: &[HirStmt<'a, 'bump>],
        expected: Option<&HirType<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        self.borrow_checker.begin_scope();
        let mut block_context = self.context.create_child_scope();

        let local_names: Vec<StrId> = body
            .iter()
            .filter_map(|s| match s {
                HirStmt::Let { name, .. } => Some(*name),
                _ => None,
            })
            .collect();

        let mut value = None;
        let last_idx = body.len().checked_sub(1);
        for (i, stmt) in body.iter().enumerate() {
            let old_context = std::mem::replace(&mut self.context, block_context);
            value = if Some(i) == last_idx {
                Some(self.check_tail_expected(stmt, expected))
            } else {
                self.check_stmt(stmt)
            };
            block_context = self.context.clone();
            self.context = old_context;

            let after_point = self.stmt_after_points.get(&Self::stmt_key(stmt)).copied();

            let dead_loans: Vec<LoanId> = self
                .loan_owners
                .iter()
                .filter(|(_, owner)| local_names.contains(owner))
                .filter(|(_, &owner)| match after_point {
                    Some(p) => !self.local_used_after(p, owner),
                    None => !body[(i + 1)..]
                        .iter()
                        .any(|s| self.stmt_references_local(s, owner)),
                })
                .map(|(&loan_id, _)| loan_id)
                .collect();
            for loan_id in dead_loans {
                self.borrow_checker.end_loan_now(loan_id);
                self.loan_owners.remove(&loan_id);
            }
        }

        self.borrow_checker.end_scope();
        // sweep anything that reached scope-end without an early kill
        self.loan_owners
            .retain(|_, owner| !local_names.contains(owner));
        for name in &local_names {
            self.local_provenance_place.remove(name);
            self.local_ref_kind.remove(name);
        }
        value
    }

    fn check_if_branches(
        &mut self,
        cond: &HirExpr<'a, 'bump>,
        then_block: &[HirStmt<'a, 'bump>],
        else_block: Option<&HirStmt<'a, 'bump>>,
        expected: Option<&HirType<'a, 'bump>>,
    ) -> Option<HirType<'a, 'bump>> {
        let snap = self.snapshot_call_loan_keys();
        let cond_type = self.check_expr(cond);
        self.end_temp_call_loans(&snap);
        if cond_type != HirType::Boolean {
            self.record(TypeErrorKind::TypeMismatch {
                expected: "bool".to_string(),
                found: self.type_to_string(&cond_type),
            });
        }

        let move_state_before = self.move_state.clone();
        let non_null_before = self.non_null_state.clone();
        let fact = self.condition_to_fact(cond);
        let non_null_fact = self.condition_to_non_null_fact(cond);
        let place_fact = self.condition_to_place_fact(cond);

        self.borrow_checker.begin_scope();
        if let Some((lhs, rhs, is_equal)) = &fact {
            if *is_equal {
                self.borrow_checker
                    .assume_equal_scoped(lhs.clone(), rhs.clone());
            } else {
                self.borrow_checker
                    .assume_not_equal_scoped(lhs.clone(), rhs.clone());
            }
        }
        if let Some((lhs, rhs, is_equal)) = place_fact {
            if !is_equal {
                self.borrow_checker.assume_places_not_equal_scoped(lhs, rhs);
            }
        }
        if let Some((root, path, holds_true)) = &non_null_fact {
            if *holds_true {
                self.mark_non_null(*root, path);
            }
        }

        let mut then_context = self.context.create_child_scope();
        let mut then_value = HirType::Void;
        if let Some((last, rest)) = then_block.split_last() {
            for stmt in rest {
                let old_context = std::mem::replace(&mut self.context, then_context);
                self.check_stmt(stmt);
                then_context = self.context.clone();
                self.context = old_context;
            }
            let old_context = std::mem::replace(&mut self.context, then_context);
            then_value = self.check_tail_expected(last, expected);
            self.context = old_context;
        }
        self.borrow_checker.end_scope();
        let then_move_state = self.move_state.clone();
        let then_non_null = self.non_null_state.clone();
        let then_diverges = matches!(then_value, HirType::Never);

        self.move_state = move_state_before.clone();
        self.non_null_state = non_null_before.clone();
        if let Some((root, path, holds_true)) = &non_null_fact {
            if !*holds_true {
                self.mark_non_null(*root, path);
            }
        }

        let mut else_value: Option<HirType<'a, 'bump>> = None;
        let mut else_diverges = false;
        if let Some(else_stmt) = else_block {
            self.borrow_checker.begin_scope();
            if let Some((lhs, rhs, is_equal)) = &fact {
                if *is_equal {
                    self.borrow_checker
                        .assume_not_equal_scoped(lhs.clone(), rhs.clone());
                } else {
                    self.borrow_checker
                        .assume_equal_scoped(lhs.clone(), rhs.clone());
                }
            }
            if let Some((lhs, rhs, is_equal)) = place_fact {
                if is_equal {
                    self.borrow_checker.assume_places_not_equal_scoped(lhs, rhs);
                }
            }
            let else_context = self.context.create_child_scope();
            let old_context = std::mem::replace(&mut self.context, else_context);
            let ev = self.check_tail_expected(else_stmt, expected);
            else_diverges = matches!(ev, HirType::Never);
            else_value = Some(ev);
            self.context = old_context;
            self.borrow_checker.end_scope();
        }
        let else_move_state = self.move_state.clone();
        let else_non_null = self.non_null_state.clone();

        self.move_state = MoveState::join(&then_move_state, &else_move_state);
        self.non_null_state = match (then_diverges, else_diverges) {
            (true, false) => else_non_null,
            (false, true) => then_non_null,
            _ => Self::join_non_null_states(&then_non_null, &else_non_null),
        };

        else_value.map(|else_ty| self.join_value_types(&[then_value, else_ty]))
    }
}

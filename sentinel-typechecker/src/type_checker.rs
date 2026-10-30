use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use ir::hir::JoinStr;

use crate::borrow_lifetime::{BorrowObligation, CalleeConcurrency};
use crate::borrowck::hir_visit::{HirVisitor, walk_expr};
use crate::dataflow::FlowState;
use crate::ensures::{EnsuresFact, InstFact, SymInit};
use crate::initialization::{BindingMode, InitNode, ModuleImports};
use crate::initialization_bitset::FastInitState;
use crate::local_table::FunctionLocalTable;
use crate::move_state::MoveState;
use crate::naming::type_to_string;
use crate::type_context::TypeContext;
use crate::{closures, statics};
use codex_dependency_graph::DepGraph;
use ir::analysis_context::CopyAnalysisCtx;
use ir::ast::FuncSafety;
use ir::attributes::{self, Arity, AttrTable};
use ir::auto_imports::AutoImportRegistry;
use ir::borrow_checker::{BorrowChecker, LoanId, PlaceId, ProvenanceId, ReadTemplate, RefTemplate};
use ir::errors::type_error::{
    TypeCheckResult, TypeError, TypeErrorKind, TypeWarning, TypeWarningKind,
};
use ir::hir::{
    Hir, HirExpr, HirFunc, HirGeneric, HirModule, HirParam, HirStmt, HirType, HirTypeKind, RefKind,
    StrId, ThisPassingKind, Visibility,
};
use ir::ir_hasher::{FxHashMap, HashSet};
use ir::nll_cfg::{Cfg, CfgBuilder, PointId};
use ir::span::SourceSpan;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub const SLICE_PRIMITIVES: &[&str] = &["write_uninit", "write_uninit_all", "get_unchecked"];

#[derive(Default)]
struct AssignedVars {
    names: Vec<StrId>,
}

impl AssignedVars {
    fn add(&mut self, n: StrId) {
        if !self.names.contains(&n) {
            self.names.push(n);
        }
    }

    fn root_ident(e: &HirExpr<'_, '_>) -> Option<StrId> {
        match e {
            HirExpr::Ident(n, _) => Some(*n),
            HirExpr::FieldAccess { object, .. }
            | HirExpr::Get { object, .. }
            | HirExpr::Index { object, .. } => Self::root_ident(object),
            _ => None,
        }
    }
}

impl<'a, 'bump> HirVisitor<'a, 'bump> for AssignedVars {
    fn visit_expr(&mut self, expr: &HirExpr<'a, 'bump>) {
        match expr {
            HirExpr::Assignment { target, .. } => {
                if let HirExpr::Ident(n, _) = target {
                    self.add(*n);
                }
            }
            HirExpr::Ref {
                expr: inner,
                ref_kind: RefKind::Unique | RefKind::Alias,
                ..
            } => {
                if let Some(n) = Self::root_ident(inner) {
                    self.add(n);
                }
            }
            _ => {}
        }
        walk_expr(self, expr);
    }
}

#[derive(Clone, Copy)]
pub struct LoanWatermark {
    pub(crate) call: usize,
    pub(crate) closure: usize,
}

pub struct Liveness {
    words: usize,
    live: Vec<u64>, // n_points * words
    idx: FxHashMap<StrId, u32>,
}

impl Liveness {
    #[inline]
    pub fn used_at_or_after(&self, p: PointId, local: StrId) -> bool {
        let Some(&b) = self.idx.get(&local) else {
            return false;
        };
        let b = b as usize;
        self.live
            .get(p.0 as usize * self.words + b / 64)
            .map_or(false, |w| w & (1u64 << (b % 64)) != 0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalSymbolId(pub u32);

/// One `unsafe impl<...> Target<...> by Send/Sync`, as written.
///
/// `unsafe impl<T: Send> Vec<T> by Send {}` is
/// `generics = [(T, ["Send"])]`, `target_args = [Generic(T)]`.
#[derive(Debug, Clone, PartialEq)]
pub struct ImplCondition<'a, 'bump> {
    /// The impl's own generic parameters and the names of their bounds.
    pub generics: Vec<(StrId, Vec<StrId>)>,
    /// The type arguments of the impl target, in order.
    pub target_args: Vec<HirType<'a, 'bump>>,
}

/// One recorded use or declaration of a symbol, for the language server.
#[derive(Clone, Copy, Debug)]
pub struct Occurrence<'a, 'bump> {
    pub span: SourceSpan<'a>,
    /// Text as written in the source. Rename and semantic tokens compare against this.
    pub name: StrId,
    /// Canonical HIR (mangled) name: the key into the context's struct / enum /
    /// interface / function tables. Equal to `name` for locals and fields.
    pub hir_name: StrId,
    pub ty: HirType<'a, 'bump>,
    pub module_idx: usize,
    pub symbol: SymbolId,
    pub is_decl: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SymbolId {
    /// let-binding, parameter, or `this`, identified by mint order, since
    /// name alone isn't unique across scopes.
    Local(LocalSymbolId),
    /// struct field, identified directly by (declaring struct HIR name, field
    /// name), already globally unique, no minting needed.
    Field {
        struct_name: StrId,
        field_name: StrId,
    },
    /// enum variant, identified by (declaring enum HIR name, variant name),
    /// globally unique like Field, no minting needed.
    Variant {
        enum_name: StrId,
        variant_name: StrId,
    },
    /// top-level declaration, same coordinate DepGraph already uses.
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

#[derive(Clone, Copy, Debug)]
pub enum InvalidationCause {
    Reallocated(StrId), // callee that may reallocate
    Dropped(StrId),     // type whose Drop ran (e.g. MutexGuard)
}

pub type NonNullMap = FxHashMap<StrId, HashSet<Vec<StrId>>>;
/// Copy-on-write: cloning is O(1); mutate through `Rc::make_mut`.
pub type NonNullState = Rc<NonNullMap>;

/// Copy-on-write like `NonNullState`.
pub type InvalidationMap = FxHashMap<ProvenanceId, InvalidationCause>;

#[derive(Clone, Copy)]
pub struct ConstDecl<'a, 'bump> {
    pub ty: HirType<'a, 'bump>,
    pub value: HirExpr<'a, 'bump>,
    pub visibility: Visibility,
    pub name: StrId,
    pub declaring_module_idx: usize,
}

pub struct TypeChecker<'a, 'bump> {
    pub(crate) context: TypeContext<'a, 'bump>,
    pub(crate) errors: Vec<TypeError<'a>>,
    pub(crate) warnings: Vec<TypeWarning<'a>>,
    pub(crate) current_span: SourceSpan<'a>,
    pub(crate) copy_analysis: Rc<RefCell<CopyAnalysisCtx<'a, 'bump>>>,
    pub(crate) move_state: MoveState,
    pub(crate) borrow_checker: BorrowChecker,
    pub(crate) this_id: StrId,
    pub(crate) suppress_errors: bool,
    pub(crate) ref_templates: FxHashMap<StrId, RefTemplate>,
    pub(crate) read_templates: FxHashMap<StrId, Vec<ReadTemplate>>,
    pub(crate) write_templates: FxHashMap<StrId, Vec<bool>>,
    pub(crate) next_symbol_id: u32,
    pub(crate) occurrences: Vec<Occurrence<'a, 'bump>>,
    pub(crate) imports_by_module: FxHashMap<usize, ModuleImports>,
    pub(crate) functions_by_module: FxHashMap<usize, HashSet<StrId>>,
    pub(crate) structs_by_module: FxHashMap<usize, HashSet<StrId>>,
    pub(crate) enums_by_module: FxHashMap<usize, HashSet<StrId>>,
    pub(crate) generic_instance_args: FxHashMap<usize, Vec<HirType<'a, 'bump>>>,
    pub(crate) loan_owners: FxHashMap<LoanId, StrId>,
    pub(crate) local_provenance_place: FxHashMap<StrId, PlaceId>,
    pub(crate) call_loans: FxHashMap<usize, LoanId>,
    pub(crate) next_opaque_id: u32,
    pub(crate) unsafe_depth: usize,
    pub(crate) auto_imports: Rc<RefCell<AutoImportRegistry>>,
    pub(crate) cfg: Cfg,
    pub(crate) stmt_points: FxHashMap<usize, PointId>,
    pub(crate) stmt_after_points: FxHashMap<usize, PointId>,
    pub(crate) point_locals_used: FxHashMap<PointId, HashSet<StrId>>,
    pub(crate) current_point: PointId,
    pub(crate) undefined_backfill: FxHashMap<usize, HirType<'a, 'bump>>,
    pub(crate) uninit_backfill: FxHashMap<usize, HirType<'a, 'bump>>,
    pub(crate) init_state: FastInitState,
    pub(crate) suppress_init_read: bool,
    pub(crate) local_ref_kind: FxHashMap<StrId, RefKind>,
    pub(crate) binding_mode_backfill: FxHashMap<usize, BindingMode>,
    pub(crate) non_null_state: NonNullState,
    pub(crate) in_place_context: bool,
    pub(crate) skip_slice_init_check: bool,
    pub(crate) closure_frames: std::cell::RefCell<Vec<closures::ClosureFrame<'bump>>>,
    pub(crate) closure_table: FxHashMap<usize, ir::hir::ClosureLowering<'a, 'bump>>,
    pub(crate) closure_loans: FxHashMap<usize, Vec<LoanId>>,
    pub(crate) call_loan_order: Vec<(usize, LoanId)>,
    pub(crate) closure_loan_order: Vec<(usize, LoanId)>,
    pub(crate) next_closure_id: u32,
    pub(crate) fn_closure_constraints: FxHashMap<StrId, HirType<'a, 'bump>>,
    pub(crate) closure_pre_subs: FxHashMap<StrId, HirType<'a, 'bump>>,
    pub(crate) closure_generic_subs: FxHashMap<StrId, HirType<'a, 'bump>>,
    pub local_provenance: FxHashMap<StrId, ProvenanceId>,
    pub(crate) current_fn: Option<StrId>,
    pub(crate) fn_invalidates: FxHashMap<StrId, Vec<Vec<StrId>>>, // paths relative to `this`
    pub(crate) fn_this_calls: FxHashMap<StrId, Vec<StrId>>,
    pub(crate) invalidated_provenances: Rc<InvalidationMap>,
    pub(crate) invalidation_cache: FxHashMap<(StrId, StrId), Vec<Vec<StrId>>>,
    pub(crate) callee_concurrency: FxHashMap<StrId, CalleeConcurrency>,
    pub(crate) borrow_obligations: Vec<BorrowObligation>,
    pub(crate) pinned_loans: HashSet<LoanId>,
    pub(crate) current_fn_bounds: FxHashMap<StrId, Vec<String>>,
    pub(crate) obligation_branches: Vec<usize>,
    pub(crate) generic_bounds: FxHashMap<StrId, Vec<String>>,
    pub(crate) module_consts: FxHashMap<usize, FxHashMap<StrId, ConstDecl<'a, 'bump>>>,
    pub(crate) const_locals: FxHashMap<StrId, i64>,
    pub(crate) const_symbols: FxHashMap<SymbolId, i64>,
    pub(crate) module_statics: FxHashMap<usize, FxHashMap<StrId, statics::StaticDecl<'a, 'bump>>>,
    pub(crate) static_symbols: FxHashMap<SymbolId, StrId>,
    pub(crate) static_summaries: FxHashMap<StrId, statics::StaticSummary>,
    pub(crate) cur_statics: Vec<StrId>,
    pub(crate) fn_static_accessed: HashSet<StrId>,
    pub(crate) fn_static_assigned: HashSet<StrId>,
    pub(crate) static_whole_target: Option<StrId>,
    pub(crate) static_assign_backfill: FxHashMap<usize, statics::StaticAssignInfo>,
    pub(crate) closure_static_uses: FxHashMap<usize, Vec<(StrId, bool)>>,
    pub(crate) tail_expr_key: Option<usize>,
    pub(crate) tail_borrows: Vec<StrId>,
    pub(crate) local_table: FunctionLocalTable,
    pub(crate) recording_effects: bool,
    pub(crate) effect_log: Vec<(PointId, crate::dataflow::Effect<'a, 'bump>)>,
    pub(crate) solved: Option<crate::dataflow::Solved>,
    pub(crate) liveness: std::cell::OnceCell<Liveness>,

    pub(crate) ensures_facts_cache: FxHashMap<StrId, Vec<EnsuresFact>>,
    pub(crate) pending_ensures: FxHashMap<usize, Vec<InstFact>>,
    pub(crate) pending_nested: FxHashMap<StrId, Vec<InstFact>>,
    pub(crate) sym_init: Vec<SymInit>,
    pub(crate) current_func: Option<HirFunc<'a, 'bump>>,
    pub(crate) pending_ret: FxHashMap<usize, Vec<crate::ensures::RetInit>>,
    /// Relational `ensures` facts of calls, keyed like `pending_ret`; applied by
    /// `apply_call_rel_facts` once the statement containing the call is known to run.
    pub(crate) pending_rel:
        FxHashMap<usize, Vec<(ir::borrow_checker::Bound, ir::borrow_checker::Bound, bool)>>,
    /// Address of the current function body's statement slice, so `check_block_with` can run
    /// the fall-through `ensures` check while the body's own facts are still in scope.
    pub(crate) fn_body_ptr: Option<usize>,
    /// Locals that can be written without a visible assignment (a `&mut` to them was taken, or
    /// a closure assigns them). No arithmetic fact about them is trusted. Per function.
    pub(crate) addr_taken: HashSet<StrId>,
    /// Nesting depth of closure bodies being checked.
    pub(crate) lambda_depth: u32,
    /// Lambdas that declare `ensures`, as functions, keyed by the address of their body.
    pub(crate) lambda_funcs: FxHashMap<usize, HirFunc<'a, 'bump>>,
    /// `let f = fn (..) ensures .. { .. };`: the binding's symbol and its lambda's function, so a
    /// later `f(..)` can assume the lambda's `ensures`. Only for immutable bindings.
    pub(crate) lambda_bindings: FxHashMap<StrId, (SymbolId, HirFunc<'a, 'bump>)>,
    pub(crate) this_struct: Option<StrId>,
}

impl<'a, 'bump> TypeChecker<'a, 'bump> {
    pub fn new(
        dep_graph: &'a RefCell<DepGraph<'a>>,
        bump: &'bump GrowableBump<'bump>,
        copy_analysis: Rc<RefCell<CopyAnalysisCtx<'a, 'bump>>>,
        string_pool: Arc<StringPool>,
        auto_imports: Rc<RefCell<AutoImportRegistry>>,
    ) -> Self {
        Self {
            this_id: StrId::from_static("this"),
            context: TypeContext::new(dep_graph, bump, string_pool),
            errors: Vec::new(),
            warnings: Vec::new(),
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
            call_loan_order: Vec::default(),
            closure_loan_order: Vec::default(),
            next_opaque_id: 0,
            unsafe_depth: 0,
            auto_imports,
            cfg: Cfg::default(),
            stmt_points: HashMap::default(),
            stmt_after_points: HashMap::default(),
            point_locals_used: HashMap::default(),
            current_point: PointId::default(),
            uninit_backfill: FxHashMap::default(),
            init_state: FastInitState::new(),
            local_ref_kind: FxHashMap::default(),
            binding_mode_backfill: FxHashMap::default(),
            invalidated_provenances: Rc::default(),
            non_null_state: NonNullState::default(),
            suppress_init_read: false,
            in_place_context: false,
            skip_slice_init_check: false,
            closure_frames: RefCell::new(Vec::default()),
            closure_table: FxHashMap::default(),
            closure_loans: FxHashMap::default(),
            next_closure_id: 0,
            fn_closure_constraints: FxHashMap::default(),
            closure_pre_subs: FxHashMap::default(),
            closure_generic_subs: FxHashMap::default(),
            local_provenance: FxHashMap::default(),
            current_fn: None,
            fn_this_calls: FxHashMap::default(),
            fn_invalidates: FxHashMap::default(),
            invalidation_cache: FxHashMap::default(),
            callee_concurrency: FxHashMap::default(),
            borrow_obligations: Vec::default(),
            pinned_loans: HashSet::default(),
            current_fn_bounds: FxHashMap::default(),
            obligation_branches: Vec::new(),
            generic_bounds: FxHashMap::default(),
            module_consts: FxHashMap::default(),
            const_locals: FxHashMap::default(),
            const_symbols: FxHashMap::default(),
            static_assign_backfill: FxHashMap::default(),
            fn_static_accessed: HashSet::default(),
            fn_static_assigned: HashSet::default(),
            static_whole_target: Option::None,
            cur_statics: Vec::default(),
            static_summaries: FxHashMap::default(),
            static_symbols: FxHashMap::default(),
            module_statics: FxHashMap::default(),
            closure_static_uses: FxHashMap::default(),
            tail_expr_key: None,
            tail_borrows: Vec::default(),
            local_table: FunctionLocalTable::new(),
            recording_effects: false,
            effect_log: Vec::default(),
            solved: None,
            liveness: std::cell::OnceCell::new(),
            ensures_facts_cache: FxHashMap::default(),
            pending_ensures: FxHashMap::default(),
            pending_nested: FxHashMap::default(),
            sym_init: Vec::new(),
            current_func: None,
            pending_ret: FxHashMap::default(),
            pending_rel: FxHashMap::default(),
            fn_body_ptr: None,
            addr_taken: HashSet::default(),
            lambda_depth: 0,
            lambda_funcs: FxHashMap::default(),
            lambda_bindings: FxHashMap::default(),
            this_struct: None,
        }
    }

    pub fn scope_end_init_snapshot(&self, name: StrId) -> Option<InitNode> {
        self.init_node(name)
    }

    pub fn uninit_ty(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.uninit_backfill.get(&Self::expr_key(expr)).copied()
    }

    fn resolve_item(
        &self,
        module_idx: usize,
        name: StrId,
        hir_name: StrId,
    ) -> Option<(usize, usize, &'static str)> {
        let dg = self.context.dep_graph.borrow();
        dg.resolve_item_in_module(module_idx, hir_name).or_else(|| {
            if hir_name != name {
                dg.resolve_item_in_module(module_idx, name)
            } else {
                None
            }
        })
    }

    pub fn record_item_occurrence_named(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        hir_name: StrId,
        ty: HirType<'a, 'bump>,
        declaring_module_idx: usize,
    ) {
        let Some((m, item_idx, tag)) = self.resolve_item(declaring_module_idx, name, hir_name)
        else {
            return;
        };
        self.occurrences.push(Occurrence {
            span,
            name,
            hir_name,
            ty,
            module_idx: self.context.current_module_idx,
            symbol: SymbolId::Item {
                module_idx: m,
                item_idx,
                tag,
            },
            is_decl: false,
        });
    }

    pub fn record_item_decl_occurrence(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        hir_name: StrId,
        ty: HirType<'a, 'bump>,
        module_idx: usize,
    ) {
        if span.line == 0 {
            return;
        }
        let Some((m, item_idx, tag)) = self.resolve_item(module_idx, name, hir_name) else {
            return;
        };
        self.occurrences.push(Occurrence {
            span,
            name,
            hir_name,
            ty,
            module_idx: self.context.current_module_idx,
            symbol: SymbolId::Item {
                module_idx: m,
                item_idx,
                tag,
            },
            is_decl: true,
        });
    }

    pub fn record_func_decl_occurrence(&mut self, func: &HirFunc<'a, 'bump>) {
        if func.name_span.line == 0 {
            return;
        }
        let ty = func.return_type.unwrap_or(HirType::void(func.name_span));
        let symbol: Option<SymbolId> = match func.impl_target {
            Some(target) => {
                self.resolve_method_sym(target, func.unmangled_name)
                    .map(|(m, i, mi)| SymbolId::Method {
                        module_idx: m,
                        item_idx: i,
                        method_idx: mi,
                    })
            }
            None => self
                .resolve_item(func.declaring_module_idx, func.unmangled_name, func.name)
                .map(|(m, i, tag)| SymbolId::Item {
                    module_idx: m,
                    item_idx: i,
                    tag,
                }),
        };
        let Some(symbol) = symbol else { return };
        self.occurrences.push(Occurrence {
            span: func.name_span,
            name: func.unmangled_name,
            hir_name: func.name,
            ty,
            module_idx: self.context.current_module_idx,
            symbol,
            is_decl: true,
        });
    }

    pub(crate) fn record_generic_occurrences(&mut self, generics: &[HirGeneric<'a, 'bump>]) {
        for g in generics {
            for c in g.constraints {
                self.record_type_occurrences(c);
            }
            if let Some(d) = g.default_type {
                self.record_type_occurrences(&d);
            }
        }
    }

    /// Same as `record_item_occurrence_named` when the source name and the
    /// HIR name are the same.
    pub fn record_item_occurrence(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        ty: HirType<'a, 'bump>,
        declaring_module_idx: usize,
    ) {
        self.record_item_occurrence_named(span, name, name, ty, declaring_module_idx);
    }

    fn resolve_method_sym(&self, target: StrId, name: StrId) -> Option<(usize, usize, usize)> {
        let dg = self.context.dep_graph.borrow();
        dg.resolve_method(target, name).or_else(|| {
            let alt = self
                .context
                .get_interface(target)
                .map(|i| i.unmangled_name)
                .or_else(|| self.context.get_struct(target).map(|s| s.unmangled_name))
                .or_else(|| self.context.get_enum(target).map(|e| e.unmangled_name))?;
            (alt != target)
                .then(|| dg.resolve_method(alt, name))
                .flatten()
        })
    }

    pub fn record_method_occurrence(
        &mut self,
        span: SourceSpan<'a>,
        name: StrId,
        ty: HirType<'a, 'bump>,
        target_type: StrId,
    ) {
        let Some((module_idx, item_idx, method_idx)) = self.resolve_method_sym(target_type, name)
        else {
            return;
        };
        self.occurrences.push(Occurrence {
            span,
            name,
            hir_name: name,
            ty,
            module_idx: self.context.current_module_idx,
            symbol: SymbolId::Method {
                module_idx,
                item_idx,
                method_idx,
            },
            is_decl: false,
        });
    }

    pub fn mint_symbol_id(&mut self) -> SymbolId {
        let id = LocalSymbolId(self.next_symbol_id);
        self.next_symbol_id += 1;
        SymbolId::Local(id)
    }

    pub fn expr_key(expr: &HirExpr<'a, 'bump>) -> usize {
        expr as *const HirExpr<'a, 'bump> as usize
    }

    pub fn record_instance_args(&mut self, expr: &HirExpr<'a, 'bump>, args: &[HirType<'a, 'bump>]) {
        self.generic_instance_args
            .insert(Self::expr_key(expr), args.to_vec());
    }

    pub fn undefined_ty(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.undefined_backfill.get(&Self::expr_key(expr)).copied()
    }

    pub fn occurrences(&self) -> &[Occurrence<'a, 'bump>] {
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

    #[inline]
    pub fn set_span(&mut self, span: SourceSpan<'a>) {
        self.current_span = span;
    }

    pub fn check_attributes(&mut self, table: &AttrTable<'a, 'bump>) {
        for target in &table.order {
            let Some(attrs) = table.map.get(target) else {
                continue;
            };
            let site = target.site();

            for (i, a) in attrs.iter().enumerate() {
                self.set_span(a.span);

                let Some(spec) = attributes::lookup(a.name.as_str()) else {
                    self.record(TypeErrorKind::Generic(format!(
                        "unknown attribute `{}`",
                        a.name
                    )));
                    continue;
                };

                if !spec.targets.contains(site) {
                    self.record(TypeErrorKind::Generic(format!(
                        "`#[{}]` cannot be applied here",
                        spec.name
                    )));
                }

                let n = a.args.len();
                let arity_err = match spec.arity {
                    Arity::None if n != 0 => Some("takes no arguments".to_string()),
                    Arity::Exactly(k) if n != k => Some(format!("expects exactly {k} argument(s)")),
                    Arity::AtMost(k) if n > k => Some(format!("takes at most {k} argument(s)")),
                    _ => None,
                };
                if let Some(msg) = arity_err {
                    self.record(TypeErrorKind::Generic(format!("`#[{}]` {msg}", spec.name)));
                }

                if !spec.repeatable && attrs[..i].iter().any(|b| b.name == a.name) {
                    self.record(TypeErrorKind::Generic(format!(
                        "duplicate `#[{}]`",
                        spec.name
                    )));
                }

                if let Some(validate) = spec.validate {
                    if let Err(msg) = validate(a) {
                        self.record(TypeErrorKind::Generic(msg));
                    }
                }
            }
        }
    }

    pub fn slice_field_owned(ty: &HirType<'a, 'bump>) -> Option<bool> {
        let inner = match ty.kind {
            HirTypeKind::Ref { inner, .. } | HirTypeKind::SafePointer { inner, .. } => *inner,
            _ => *ty,
        };
        match inner.kind {
            HirTypeKind::Slice(_) => Some(false),
            HirTypeKind::OwnedPointer { inner, .. } => match inner.kind {
                HirTypeKind::Slice(_) => Some(true),
                _ => None,
            },
            _ => None,
        }
    }

    pub fn record(&mut self, kind: TypeErrorKind) {
        if self.suppress_errors {
            return;
        }
        self.errors.push(kind.at(self.current_span));
    }

    pub fn recover<T>(&mut self, result: TypeCheckResult<'a, T>, fallback: T) -> T {
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

    pub fn with_suppressed_errors<F: FnOnce(&mut Self)>(&mut self, f: F) {
        let prev = self.suppress_errors;
        self.suppress_errors = true;
        f(self);
        self.suppress_errors = prev;
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
                Hir::Const(c) | Hir::Stmt(HirStmt::Const(c)) => {
                    self.module_consts.entry(module_idx).or_default().insert(
                        c.name,
                        ConstDecl {
                            ty: c.ty,
                            visibility: c.visibility,
                            declaring_module_idx: self.context.current_module_idx,
                            name: c.name,
                            value: c.value,
                        },
                    );
                }
                Hir::Stmt(HirStmt::Let {
                    name,
                    ty,
                    value,
                    is_static: true,
                    span,
                    visibility,
                    ..
                }) => {
                    self.register_static(module_idx, *name, *ty, value, *visibility, *span);
                }
                Hir::Impl(i) => {
                    let target = i.target;

                    if let Some(interface) = i.interface {
                        let generics: Vec<(StrId, Vec<StrId>)> = i
                            .generics
                            .unwrap_or(&[])
                            .iter()
                            .map(|g| {
                                if let Some(def_ty) = g.default_type {
                                    self.record_type_occurrences(&def_ty);
                                }

                                let constraints = g
                                    .constraints
                                    .iter()
                                    .map(|c| {
                                        self.record_type_occurrences(c);
                                        type_to_string(c, &self.context.string_pool)
                                    })
                                    .collect();
                                (g.name, constraints)
                            })
                            .collect();

                        self.register_interface_impl(
                            target.as_str(),
                            interface.as_str(),
                            i.is_unsafe,
                            generics,
                            i.target_generics.unwrap_or_default(),
                        );
                    }
                    if let Some(methods) = i.methods {
                        let table = std::sync::Arc::make_mut(&mut self.context.globals)
                            .type_methods
                            .entry(target)
                            .or_default();
                        for func in methods {
                            if table.methods.iter().any(|(name, _)| *name == func.name) {
                                self.current_span = func.span;
                                if self.suppress_errors {
                                    continue;
                                }
                                self.errors.push(TypeErrorKind::Generic(format!(
                                    "function `{}` is already declared in this module with the same signature",
                                    func.unmangled_name
                                )).at(self.current_span));
                            }
                            table.insert(func.unmangled_name, *func);
                        }
                        // Why can't I just do this in the exact same loop above?
                        // I asked rust, it told me "cannot borrow *self as mutable more than once at a time 🤓🤓"
                        // because I couldn't register callee concurrency while holding self.context.type_methods
                        // This is not the most efficient solution but it gets the job done
                        //
                        // Update: same issue happened for check_return_type_visibility
                        for func in methods {
                            self.check_return_type_visibility(func, func.span);
                            self.register_callee_concurrency(func);
                        }
                    }
                    if let Some(interface) = i.interface {
                        self.context
                            .add_struct_interface(&target, interface.to_string());
                    }
                }
                Hir::Interface(i) => {
                    self.context.add_interface(module_idx, i.name, **i);
                }
                Hir::Enum(e) => {
                    self.context.add_enum(module_idx, e.name, **e);
                    self.enums_by_module
                        .entry(module_idx)
                        .or_default()
                        .insert(e.name);
                }
                Hir::Func(f) => {
                    self.register_callee_concurrency(f);
                    if self
                        .functions_by_module
                        .get(&module_idx)
                        .is_some_and(|s| s.contains(&f.name))
                    {
                        self.set_span(f.span);
                        self.record(TypeErrorKind::Generic(format!(
                            "function `{}` is already declared in this module with the same signature",
                            f.unmangled_name
                        )));
                    }

                    if let Some(params) = f.params
                        && let Some(this_param) = params.first()
                    {
                        if matches!(this_param, HirParam::This { .. }) {
                            self.record(TypeErrorKind::IllegalThisParam {
                                func_name: f.unmangled_name.to_string(),
                            });
                        }
                    }

                    if f.name == f.unmangled_name
                        || self
                            .context
                            .get_module_function(module_idx, f.unmangled_name)
                            .is_none()
                    {
                        self.context.add_function(module_idx, f.unmangled_name, **f);
                    }
                    if f.name != f.unmangled_name {
                        self.context.add_function(module_idx, f.name, **f);
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
            modules: HashSet::default(),
            module_aliases: HashMap::default(),
            wildcard: Vec::new(),
        };
        for import_path in module.imports {
            let Some(target_module) = self
                .context
                .dep_graph
                .borrow()
                .resolve_module_path(import_path.path)
            else {
                let path_str = import_path.path.join("::");
                self.record(TypeErrorKind::Generic(format!(
                    "cannot resolve imported module `{}`",
                    path_str
                )));
                continue;
            };
            match import_path.member {
                Some(name) => {
                    let real_module = self
                        .context
                        .dep_graph
                        .borrow()
                        .canonical_member_module(target_module, name);
                    imports.named.insert(name, real_module);
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

        let static_auto_paths: Vec<Vec<StrId>> = self
            .auto_imports
            .borrow()
            .paths()
            .map(|p| p.iter().map(|s| StrId::from_static(s)).collect())
            .collect();
        let impl_auto_paths = self.context.dep_graph.borrow().auto_import_packages();

        for segments in static_auto_paths.into_iter().chain(impl_auto_paths) {
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
                            .unwrap_or_default();
                        let new_pkg = self
                            .context
                            .dep_graph
                            .borrow()
                            .get_module_package(target_module)
                            .unwrap_or_default();
                        if self.suppress_errors {
                            continue;
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
        self.occurrences.retain(|o| o.module_idx != module_idx);
        self.context.current_module_idx = module_idx;

        // Declaration-side occurrences for struct fields, and the types they mention.
        for item in module.items {
            match item {
                Hir::Struct(s) => {
                    self.this_struct = Some(s.name);
                    for f in s.fields {
                        self.occurrences.push(Occurrence {
                            span: f.span,
                            name: f.name,
                            hir_name: f.name,
                            ty: f.field_type,
                            module_idx,
                            symbol: SymbolId::Field {
                                struct_name: s.name,
                                field_name: f.name,
                            },
                            is_decl: true,
                        });
                        self.record_type_occurrences(&f.field_type);
                    }
                    self.this_struct = None;
                    self.record_item_decl_occurrence(
                        s.name_span,
                        s.unmangled_name,
                        s.name,
                        HirType::struct_type(s.name, s.unmangled_name, &[], &[], s.name_span),
                        module_idx,
                    );
                    self.record_generic_occurrences(s.generics.unwrap_or(&[]));
                }
                Hir::Enum(e) => {
                    let ety = HirType::new(
                        HirTypeKind::Enum {
                            name: e.name,
                            unmangled_name: e.unmangled_name,
                            variants: e.variants,
                            type_args: &[],
                        },
                        e.name_span,
                    );
                    self.record_item_decl_occurrence(
                        e.name_span,
                        e.unmangled_name,
                        e.name,
                        ety,
                        module_idx,
                    );
                    self.record_generic_occurrences(e.generics.unwrap_or(&[]));
                    for v in e.variants {
                        if v.span.line != 0 {
                            self.occurrences.push(Occurrence {
                                span: v.span,
                                name: v.name,
                                hir_name: v.name,
                                ty: ety,
                                module_idx,
                                symbol: SymbolId::Variant {
                                    enum_name: e.name,
                                    variant_name: v.name,
                                },
                                is_decl: true,
                            });
                        }
                        for f in v.fields {
                            self.record_type_occurrences(&f.field_type);
                        }
                    }
                }
                Hir::Interface(i) => {
                    let ity = HirType::new(
                        HirTypeKind::DynInterface {
                            name: i.name,
                            unmangled_name: i.unmangled_name,
                            type_args: &[],
                        },
                        i.name_span,
                    );
                    self.record_item_decl_occurrence(
                        i.name_span,
                        i.unmangled_name,
                        i.name,
                        ity,
                        module_idx,
                    );
                    self.record_generic_occurrences(i.generics.unwrap_or(&[]));
                    for m in i.methods.unwrap_or(&[]) {
                        self.record_func_decl_occurrence(m);
                        for p in m.params.unwrap_or(&[]) {
                            if let HirParam::Normal { param_type, .. } = p {
                                self.record_type_occurrences(param_type);
                            }
                        }
                        if let Some(r) = m.return_type {
                            self.record_type_occurrences(&r);
                        }
                    }
                }
                Hir::Impl(i) => {
                    self.record_generic_occurrences(i.generics.unwrap_or(&[]));
                    for t in i.target_generics.unwrap_or(&[]) {
                        self.record_type_occurrences(t);
                    }
                    for t in i.interface_generics.unwrap_or(&[]) {
                        self.record_type_occurrences(t);
                    }
                    self.record_type_name_use(i.target_span, i.target);
                    if let Some(iface) = i.interface {
                        self.record_type_name_use(i.interface_span, iface);
                    }
                }
                Hir::Stmt(HirStmt::Let {
                    name,
                    ty,
                    span,
                    is_static: true,
                    inferred_ty,
                    ..
                }) => {
                    if !*inferred_ty {
                        self.record_type_occurrences(ty);
                    }
                    self.record_item_decl_occurrence(*span, *name, *name, *ty, module_idx);
                }
                Hir::Const(c) | Hir::Stmt(HirStmt::Const(c)) => self.record_type_occurrences(&c.ty),
                _ => {}
            }
        }

        let mut funcs: Vec<(&HirFunc<'a, 'bump>, Option<&'bump [HirGeneric<'a, 'bump>]>)> =
            Vec::new();
        for item in module.items {
            match item {
                Hir::Func(f) => funcs.push((f, None)),
                Hir::Impl(i) => {
                    if let Some(ms) = i.methods {
                        for f in ms {
                            funcs.push((f, i.generics));
                        }
                    }
                }
                _ => {}
            }
        }
        let mut name_to_func: FxHashMap<StrId, usize> = FxHashMap::default();
        name_to_func.reserve(funcs.len());
        for (j, (f, _)) in funcs.iter().enumerate() {
            name_to_func.entry(f.unmangled_name).or_insert(j);
        }
        for (j, (f, _)) in funcs.iter().enumerate() {
            name_to_func.insert(f.name, j);
        }
        let used: Vec<HashSet<StrId>> = funcs.iter().map(|(f, _)| self.idents_used_in(f)).collect();
        let mut done = vec![false; funcs.len()];
        let mut order = Vec::with_capacity(funcs.len());
        fn visit(
            i: usize,
            used: &[HashSet<StrId>],
            name_to_func: &FxHashMap<StrId, usize>,
            done: &mut [bool],
            order: &mut Vec<usize>,
        ) {
            if done[i] {
                return;
            }
            done[i] = true;
            for name in &used[i] {
                if let Some(&j) = name_to_func.get(name) {
                    visit(j, used, name_to_func, done, order);
                }
            }
            order.push(i);
        }
        for i in 0..funcs.len() {
            visit(i, &used, &name_to_func, &mut done, &mut order);
        }
        for i in order {
            let (f, g) = funcs[i];
            self.check_function_in(f, g);
        }

        self.context.current_module_idx = module_idx;

        for item in module.items {
            if let Hir::Struct(ty_struct) = item {
                let Some(struct_interfaces) = self.context.struct_interfaces.get(ty_struct.name)
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
        self.propagate_invalidations();
    }

    pub fn check_function(&mut self, func: &HirFunc<'a, 'bump>) {
        self.check_function_in(func, None);
    }

    pub fn check_function_in(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        impl_generics: Option<&'bump [HirGeneric<'a, 'bump>]>,
    ) {
        if func.function_metadata.func_safety == FuncSafety::Unsafe {
            self.unsafe_depth += 1;
        }

        self.generic_bounds.clear();
        self.this_struct = func.impl_target;

        self.record_func_decl_occurrence(func);
        let own_generics: Vec<HirGeneric<'a, 'bump>> = func
            .generics
            .unwrap_or(&[])
            .iter()
            .filter(|g| {
                !impl_generics.unwrap_or(&[]).iter().any(|ig| {
                    ig.name == g.name
                        && ig.span.line == g.span.line
                        && ig.span.column == g.span.column
                })
            })
            .copied()
            .collect();
        self.record_generic_occurrences(&own_generics);

        self.fn_closure_constraints.clear();
        for g in impl_generics
            .unwrap_or(&[])
            .iter()
            .chain(func.generics.unwrap_or(&[]).iter())
        {
            let mut names = Vec::new();
            for c in g.constraints.iter() {
                Self::collect_bound_names(c, &mut names);
            }
            self.generic_bounds.insert(g.name, names);
            if let Some(c) = g
                .constraints
                .iter()
                .find(|c| matches!(c.kind, HirTypeKind::Lambda { .. }))
            {
                self.fn_closure_constraints.insert(g.name, *c);
            }
        }

        self.this_struct = None;
        self.current_fn = Some(func.name);
        self.current_func = Some(*func);
        let mut func_context = self.context.create_child_scope();

        self.local_provenance.clear();
        self.local_provenance_place.clear();
        self.call_loan_order.clear();
        self.invalidated_provenances = Rc::default();
        self.sym_init.clear();
        self.pending_ensures.clear();
        self.pending_nested.clear();
        self.pending_rel.clear();
        self.fn_body_ptr = None;
        self.addr_taken.clear();
        self.lambda_depth = 0;
        self.lambda_funcs.clear();
        self.lambda_bindings.clear();
        self.borrow_checker = BorrowChecker::new();
        self.borrow_checker.begin_scope();

        self.move_state = MoveState::new();
        self.init_state = FastInitState::new();
        self.non_null_state = NonNullState::default();
        self.local_ref_kind = FxHashMap::default();
        self.const_locals.clear();

        self.loan_owners.clear();
        self.pinned_loans.clear();
        self.call_loans.clear();
        self.closure_loans.clear();
        self.closure_loan_order.clear();

        self.effect_log.clear();
        self.solved = None;
        self.recording_effects = false;

        self.begin_concurrency_function(func);

        if let Some(params) = func.params {
            for param in params.iter() {
                match param {
                    HirParam::Normal {
                        name,
                        param_type,
                        span,
                        ..
                    } => {
                        let symbol_id = self.mint_symbol_id();
                        if func_context.get_variable(name).is_some() {
                            self.record(TypeErrorKind::VariableAlreadyExists {
                                var_name: name.to_string(),
                            });
                        }
                        func_context.add_variable(name, *param_type, symbol_id);
                        self.borrow_checker.declare_local(*name);

                        // Params are initialized on entry.
                        let local_id = self.local_table.register(*name);
                        self.init_state.mark_whole_init(local_id);
                        self.occurrences.push(Occurrence {
                            span: *span,
                            name: *name,
                            hir_name: *name,
                            ty: *param_type,
                            module_idx: self.context.current_module_idx,
                            symbol: symbol_id,
                            is_decl: true,
                        });
                        self.record_type_occurrences(param_type);
                    }
                    HirParam::This { kind, span, .. } => {
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

                        let self_ty = self.this_type_for_func(func, *span);
                        let rk = match kind {
                            ThisPassingKind::RefConst => Some(RefKind::Shared),
                            ThisPassingKind::RefMut => Some(RefKind::Unique),
                            ThisPassingKind::RefAlias => Some(RefKind::Alias),
                            _ => None,
                        };
                        let self_ty = match rk {
                            Some(ref_kind) => HirType {
                                kind: HirTypeKind::Ref {
                                    inner: self.context.bump.alloc_value(self_ty),
                                    ref_kind,
                                    provenance: None,
                                },
                                span: *span,
                            },
                            None => self_ty,
                        };
                        let symbol_id = self.mint_symbol_id();
                        func_context.add_variable("this", self_ty, symbol_id);
                        self.borrow_checker.declare_local(self.this_id);

                        let this_local = self.local_table.register(self.this_id);
                        self.init_state.mark_whole_init(this_local);
                        // `self_ty` is deliberately NOT passed to `record_type_occurrences`:
                        // it carries the span of the `this` token, not of a type name.
                        self.occurrences.push(Occurrence {
                            span: *span,
                            name: self.this_id,
                            hir_name: self.this_id,
                            ty: self_ty,
                            module_idx: self.context.current_module_idx,
                            symbol: symbol_id,
                            is_decl: true,
                        });
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

        if let Some(ret) = func.return_type {
            self.record_type_occurrences(&ret);
        }

        self.declare_statics_in_scope(&mut func_context);

        func_context.current_return_type = func.return_type;

        let outer = std::mem::replace(&mut self.context, func_context);
        let _ = self.analyze_ensures(func, true);
        func_context = std::mem::replace(&mut self.context, outer);

        if let Some(body) = func.body {
            let (cfg, points) = CfgBuilder::new().build(&body);
            self.cfg = cfg;
            self.liveness = std::cell::OnceCell::new();
            self.stmt_points = points.stmt_points;
            self.stmt_after_points = points.stmt_after_points;
            self.point_locals_used = FxHashMap::default();

            self.current_point = self.cfg.entry.unwrap_or_default();
            self.collect_locals_used_stmt(&body);
            self.current_point = self.cfg.entry.unwrap_or_default();

            let old_context = std::mem::replace(&mut self.context, func_context);

            match &body {
                HirStmt::Block { body: stmts, .. } if !stmts.is_empty() => {
                    self.fn_body_ptr = Some(stmts.as_ptr() as usize);
                }
                HirStmt::Block { .. } => self.check_ensures_at_fallthrough(func),
                _ => {}
            }

            self.check_stmt(&body);
            self.fn_body_ptr = None;
            self.context = old_context;
        }
        self.finish_function_obligations();

        self.check_return_provenance(func);

        self.borrow_checker.end_scope();
        self.finish_static_summary(func);
        self.current_fn = None;
        self.current_func = None;
        if func.function_metadata.func_safety == FuncSafety::Unsafe {
            self.unsafe_depth -= 1;
        }
    }

    pub fn this_type_for_func(
        &self,
        func: &HirFunc<'a, 'bump>,
        span: SourceSpan<'a>,
    ) -> HirType<'a, 'bump> {
        let Some(target) = func.impl_target else {
            return HirType::this(span);
        };

        if let Some(def) = self.context.get_struct(target) {
            let type_args: Vec<HirType<'a, 'bump>> = def
                .generics
                .unwrap_or(&[])
                .iter()
                .map(|g| HirType::generic(g.name, g.span))
                .collect();
            let field_types: Vec<HirType<'a, 'bump>> =
                def.fields.iter().map(|f| f.field_type).collect();
            return HirType {
                kind: HirTypeKind::Struct {
                    name: target,
                    unmangled_name: def.unmangled_name,
                    field_types: self.context.bump.alloc_slice(&field_types),
                    type_args: self.context.bump.alloc_slice_copy(&type_args),
                },
                span,
            };
        }

        if let Some(interface) = self.context.get_interface(target) {
            return HirType {
                kind: HirTypeKind::DynInterface {
                    name: target,
                    unmangled_name: interface.unmangled_name,
                    type_args: &[],
                },
                span,
            };
        }

        if let Some(def) = self.context.get_enum(target) {
            let type_args: Vec<_> = def
                .generics
                .unwrap_or(&[])
                .iter()
                .map(|g| HirType::generic(g.name, g.span))
                .collect();
            return HirType {
                kind: HirTypeKind::Enum {
                    name: target,
                    unmangled_name: def.unmangled_name,
                    type_args: self.context.bump.alloc_slice_copy(&type_args),
                    variants: def.variants,
                },
                span,
            };
        }
        if let Some(p) = Self::primitive_by_name(target.as_str(), span) {
            return p;
        }

        HirType::this(span)
    }

    fn primitive_by_name(n: &str, span: SourceSpan<'a>) -> Option<HirType<'a, 'bump>> {
        Some(match n {
            "i8" => HirType::i8(span),
            "i16" => HirType::i16(span),
            "i32" => HirType::i32(span),
            "i64" => HirType::i64(span),
            "i128" => HirType::i128(span),
            "u8" => HirType::u8(span),
            "u16" => HirType::u16(span),
            "u32" => HirType::u32(span),
            "u64" => HirType::u64(span),
            "u128" => HirType::u128(span),
            "usize" => HirType::usize(span),
            "isize" => HirType::isize(span),
            "f32" => HirType::f32(span),
            "f64" => HirType::f64(span),
            "bool" => HirType::boolean(span),
            "char" => HirType::char(span),
            "str" => HirType::string(span),
            _ => return None,
        })
    }

    pub fn stmt_key(stmt: &HirStmt<'a, 'bump>) -> usize {
        stmt as *const HirStmt<'a, 'bump> as usize
    }

    pub fn set_point(&mut self, stmt: &HirStmt<'a, 'bump>) {
        if let Some(&point) = self.stmt_points.get(&Self::stmt_key(stmt)) {
            self.current_point = point;
        }
    }

    pub fn check_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        let r = self.check_stmt_inner(stmt);
        if matches!(
            stmt,
            HirStmt::Expr(_) | HirStmt::Return(..) | HirStmt::Break(..)
        ) {
            self.finish_statement_obligations();
        }
        if matches!(stmt, HirStmt::Return(..)) {
            self.check_obligations_at_return();
        }
        r
    }

    fn prepare_loop_flow(&mut self, loop_stmt: &HirStmt<'a, 'bump>) {
        if self.recording_effects || self.context.in_loop {
            return; // nested loops are covered by the outermost loop's solve
        }
        let entry = FlowState {
            mv: self.move_state.clone(),
            init: self.init_state.clone(),
        };
        let saved_point = self.current_point;
        let occ_len = self.occurrences.len();

        let mut child = self.context.create_child_scope();
        child.current_return_type = self.context.current_return_type;
        let saved_ctx = std::mem::replace(&mut self.context, child);

        self.effect_log.clear();
        self.recording_effects = true;
        self.with_suppressed_errors(|t| {
            t.check_stmt(loop_stmt);
        });
        self.recording_effects = false;

        self.occurrences.truncate(occ_len);
        self.context = saved_ctx;
        self.move_state = entry.mv.clone();
        self.init_state = entry.init.clone();
        self.current_point = saved_point;
        self.solved = Some(self.solve_flow(entry));
    }

    pub fn check_stmt_inner(&mut self, stmt: &HirStmt<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.set_point(stmt);
        match stmt {
            HirStmt::Let {
                name,
                ty,
                value,
                mutable,
                span,
                is_static: _,
                catch_pattern: _,
                inferred_ty,
                manual: _,
                visibility: _,
            } => {
                self.set_span(*span);
                let r = self.check_let_stmt(name, ty, value, mutable, span, *inferred_ty);
                if let Some(call) = Self::scrutinee_call_expr(value) {
                    self.apply_ret_ensures(*name, call);
                }
                match value {
                    HirExpr::Number(n, _) => {
                        self.const_locals.insert(*name, *n);
                    }
                    _ => {
                        self.const_locals.remove(name);
                    }
                }
                // immutable binding with a compile-time-constant initialiser
                if !*mutable {
                    if let Some(c) = self.const_eval(value) {
                        if let Some((sym, _)) = self.context.get_variable(name) {
                            self.const_symbols.insert(sym, c);
                        }
                    }
                }
                r
            }
            HirStmt::Return(expr, span) => {
                self.set_span(*span);
                let skip_read = match self.current_func {
                    Some(f) => self.check_return_against_ensures(&f, *expr),
                    None => false,
                };
                let prev = self.suppress_init_read;
                if skip_read {
                    self.suppress_init_read = true;
                }
                let r = self.check_return_stmt(expr, *span);
                self.suppress_init_read = prev;
                r
            }
            HirStmt::Expr(e) => self.check_expr_stmt(e),
            HirStmt::If {
                cond,
                then_block,
                else_block,
                span,
            } => {
                self.set_span(*span);
                self.check_if_branches(cond, *then_block, *else_block, None)
            }
            // TODO: add spans to While, For and Const stmts
            HirStmt::While { cond, body } => {
                self.kill_facts_for_loop(stmt);
                let fills = self.detect_counted_fills(cond, body);
                self.prepare_loop_flow(stmt);
                let r = self.check_while_stmt(cond, body);
                if !self.context.in_loop {
                    self.solved = None;
                }
                self.apply_counted_fills(fills);
                r
            }
            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => {
                self.kill_facts_for_loop(stmt);
                self.prepare_loop_flow(stmt);
                let r = self.check_for_stmt(init, condition, increment, body);
                if !self.context.in_loop {
                    self.solved = None;
                }
                r
            }
            HirStmt::Block { body, span } => {
                self.set_span(*span);
                self.check_block_stmt(body)
            }
            HirStmt::Break(expr, span) => {
                self.set_span(*span);
                self.check_break_stmt(expr, *span)
            }
            HirStmt::Continue(span) => {
                self.set_span(*span);
                if !self.context.in_loop {
                    self.record(TypeErrorKind::ContinueOutsideLoop);
                }
                Some(HirType::never(*span))
            }
            HirStmt::Const(const_stmt) => self.check_const_stmt(const_stmt),
            HirStmt::Match { expr, arms, span } => {
                self.set_span(*span);
                Some(self.check_match_arms(expr, arms, None))
            }
            HirStmt::UnsafeBlock { body } => self.check_unsafe_stmt(body),
            HirStmt::Defer(hir_stmt) => self.check_stmt(hir_stmt),
            HirStmt::Import(path, span) => {
                self.set_span(*span);
                self.check_import_stmt(path)
            }
            HirStmt::Package(path, span) => {
                self.set_span(*span);
                self.check_package_stmt(path)
            }
        }
    }

    pub fn check_expr_expected(
        &mut self,
        expr: &HirExpr<'a, 'bump>,
        expected: &HirType<'a, 'bump>,
    ) -> HirType<'a, 'bump> {
        if matches!(expr, HirExpr::Ref { .. }) {
            return self.check_expr(expr);
        }
        match expr {
            HirExpr::Call {
                callee,
                args,
                span,
                type_args,
            } => {
                let t = self.check_call_expr(expr, callee, args, span, *type_args, Some(expected));
                self.apply_static_call_effects(callee, *span);
                self.stash_lambda_call(expr);
                t
            }
            HirExpr::Number(_, span) if self.is_integer(Self::peel_nullable(expected)) => {
                self.set_span(*span);
                *Self::peel_nullable(expected)
            }
            HirExpr::Decimal(_, span)
                if matches!(
                    Self::peel_nullable(expected).kind,
                    HirTypeKind::F32 | HirTypeKind::F64
                ) =>
            {
                self.set_span(*span);
                *Self::peel_nullable(expected)
            }
            HirExpr::Undefined { span, ty } if ty.kind == HirTypeKind::Unknown => {
                self.set_span(*span);
                if !self.is_zeroable(expected) {
                    self.record(TypeErrorKind::Generic(format!(
                    "`undefined` cannot be used for type `{}`: it cannot be safely zero-initialized",
                    expected
                )));
                    return HirType::unknown(*span);
                }
                self.undefined_backfill
                    .insert(Self::expr_key(expr), *expected);
                *expected
            }
            HirExpr::StructInit {
                name,
                args,
                type_args,
                span,
            } => {
                self.set_span(*span);
                self.check_struct_init_expr(name, args, type_args, Some(expected))
            }
            HirExpr::Uninit { span, ty } if ty.kind == HirTypeKind::Unknown => {
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
                variant_span,
            } => self.check_enum_init_expr(
                expr,
                enum_name,
                variant,
                args,
                type_args,
                *span,
                Some(expected),
                *variant_span,
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
                    .unwrap_or(HirType::void(*span))
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
                    .unwrap_or(HirType::void(*span));
                if *is_unsafe {
                    self.unsafe_depth -= 1;
                }
                ret_ty
            }

            HirExpr::Lambda { .. } => match expected.kind {
                HirTypeKind::Lambda { .. } => self.check_lambda(expr, Some(*expected), false),
                _ => self.check_lambda(expr, None, false),
            },

            HirExpr::Binary {
                left,
                op,
                right,
                span,
            } if self.is_numeric(Self::peel_nullable(expected))
                && matches!(
                    op,
                    ir::hir::Operator::Add
                        | ir::hir::Operator::Subtract
                        | ir::hir::Operator::Multiply
                        | ir::hir::Operator::Divide
                        | ir::hir::Operator::Modulo
                        | ir::hir::Operator::BitAnd
                        | ir::hir::Operator::BitOr
                        | ir::hir::Operator::BitXor
                ) =>
            {
                self.set_span(*span);
                let exp = *Self::peel_nullable(expected);
                let lt = self.check_expr_expected(left, &exp);
                let rt = self.check_expr_expected(right, &exp);
                let r = self.check_binary_op(&lt, op, &rt);
                self.recover(r, HirType::unknown(*span))
            }

            _ => self.check_expr(expr),
        }
    }

    pub fn check_expr_as_place(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        let prev = self.in_place_context;
        self.in_place_context = true;
        let ty = self.check_expr_as_place_inner(expr);
        self.in_place_context = prev;
        ty
    }

    pub fn check_expr_as_place_inner(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        match expr {
            HirExpr::Ident(name, span) => {
                let is_static = self.is_static_ident(*name);
                let (symbol_id, ty) = match self.context.get_variable(name) {
                    Some(ty) => ty,
                    None => {
                        self.record(TypeErrorKind::UndefinedVariable(name.to_string()));
                        (
                            SymbolId::Local(LocalSymbolId(u32::MAX)),
                            HirType::unknown(*span),
                        )
                    }
                };
                self.point_locals_used
                    .entry(self.current_point)
                    .or_default()
                    .insert(*name);
                if is_static {
                    if let Some((decl, owner)) = self.lookup_module_static_with_owner(*name) {
                        self.record_item_occurrence(*span, *name, decl.ty, owner);
                    }
                } else {
                    self.occurrences.push(Occurrence {
                        span: *span,
                        name: *name,
                        hir_name: *name,
                        ty,
                        module_idx: self.context.current_module_idx,
                        symbol: symbol_id,
                        is_decl: false,
                    });
                }
                self.note_static_access(*name);
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
                let index_ty = self.check_expr_expected(index, &HirType::usize(*span));
                self.recover(self.types_compatible(&HirType::usize(*span), &index_ty), ());
                match object_ty.kind {
                    HirTypeKind::SafePointer { inner, .. }
                    | HirTypeKind::UnsafePointer { inner, .. } => {
                        if !self.in_unsafe() {
                            self.record(TypeErrorKind::Generic(
                                "indexing a raw/unsafe pointer requires an unsafe block"
                                    .to_string(),
                            ));
                        }
                        *inner
                    }
                    _ => match Self::strip_ref(&object_ty).kind {
                        HirTypeKind::Array(inner, _) => *inner,
                        HirTypeKind::Slice(inner) => *inner,
                        _ => {
                            self.record(TypeErrorKind::Generic(format!(
                                "cannot index type `{}`",
                                object_ty
                            )));
                            HirType::unknown(*span)
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
                if !skip_init && matches!(Self::strip_ref(&object_ty).kind, HirTypeKind::Array(..))
                {
                    self.check_slice_range_init(object, start, end);
                }
                match Self::strip_ref(&object_ty).kind {
                    HirTypeKind::Array(inner, _) | HirTypeKind::Slice(inner) => {
                        HirType::slice(inner, *span)
                    }
                    _ => {
                        self.record(TypeErrorKind::Generic(format!(
                            "cannot slice type `{}`",
                            object_ty
                        )));
                        HirType::unknown(*span)
                    }
                }
            }
            _ => self.check_expr(expr),
        }
    }

    pub fn check_expr(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
        let span = expr.get_span();
        self.set_span(span);
        match expr {
            HirExpr::Number(_, _) => HirType::i64(span),
            HirExpr::Null(_) => HirType::null(span),
            HirExpr::Decimal(_, _) => HirType::f64(span),
            HirExpr::Boolean(_, _) => HirType::boolean(span),
            HirExpr::String(_, _) => HirType::string(span),
            HirExpr::Uninit { span, ty } => {
                self.set_span(*span);
                self.check_uninit_value(ty)
            }
            HirExpr::Undefined { span, ty } => {
                self.set_span(*span);
                self.check_zeroed_value(ty)
            }

            HirExpr::Ident(name, span) => self.check_ident_expr(name, span),
            HirExpr::Tuple(exprs, _span) => {
                let mut types = Vec::new();
                for e in *exprs {
                    types.push(self.check_expr(e));
                }
                HirType::tuple(self.context.bump.alloc_slice_copy(types.as_slice()), span)
            }
            HirExpr::Binary {
                left,
                op,
                right,
                span,
            } => {
                self.set_span(*span);
                self.check_binary_expr(left, op, right)
            }
            HirExpr::Intrinsic {
                kind,
                type_args,
                args,
                span,
            } => {
                self.set_span(*span);

                self.check_intrinsic_expr(expr, kind, type_args, args)
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
                self.check_stmt(if_stmt).unwrap_or(HirType::void(*span))
            }

            HirExpr::Match { expr, arms, span } => {
                self.set_span(*span);
                self.check_match_expr(expr, arms)
            }

            HirExpr::Block {
                body,
                is_unsafe,
                span,
            } => {
                self.set_span(*span);
                self.check_block_expr(body, is_unsafe)
            }

            HirExpr::Range {
                start,
                end,
                inclusive,
                span,
            } => {
                self.set_span(*span);
                self.check_range_expr(start, end, inclusive)
            }

            HirExpr::Slice {
                object,
                start,
                end,
                inclusive: _,
                span,
            } => {
                self.set_span(*span);
                self.check_slice_expr(object, start, end)
            }
            HirExpr::Call {
                callee,
                args,
                span,
                type_args,
            } => {
                let t = self.check_call_expr(expr, callee, args, span, *type_args, None);
                self.apply_static_call_effects(callee, *span);
                self.stash_lambda_call(expr);
                t
            }
            HirExpr::FieldAccess {
                object,
                field,
                span,
            } => {
                self.set_span(*span);
                self.check_field_access_expr(object, *field)
            }
            HirExpr::StructInit {
                name,
                args,
                span,
                type_args,
            } => {
                self.set_span(*span);
                self.check_struct_init_expr(name, args, type_args, None)
            }
            HirExpr::InterfaceCall {
                callee,
                interface,
                args,
                span,
            } => {
                self.set_span(*span);
                self.check_interface_call_expr(expr, callee, interface, args, *span)
            }

            HirExpr::Assignment {
                target,
                op,
                value,
                span,
            } => {
                self.set_span(*span);
                self.static_assignment_pre(target, op);
                let t = self.check_assignment_expr(target, op, value);
                self.static_assignment_post(value);
                t
            }
            HirExpr::InterpolatedString(parts) => {
                for part in *parts {
                    if let ir::hir::InterpolationPart::Expr(e) = part {
                        self.check_expr(e);
                    }
                }
                HirType::string(span)
            }
            HirExpr::EnumInit {
                enum_name,
                variant,
                args,
                type_args,
                span,
                variant_span,
            } => self.check_enum_init_expr(
                expr,
                enum_name,
                variant,
                args,
                type_args,
                *span,
                None,
                *variant_span,
            ),
            HirExpr::ExprList { list, span } => {
                self.set_span(*span);
                let mut last = HirType::void(*span);
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
                self.check_field_access_expr(object, *field)
            }
            HirExpr::Comparison {
                left,
                op,
                right,
                span,
            } => {
                self.set_span(*span);
                self.check_comparison_expr(left, op, right)
            }
            HirExpr::Deref { expr, span } => {
                self.set_span(*span);
                self.check_deref_expr(expr)
            }
            HirExpr::Ref {
                expr,
                ref_kind,
                span,
            } => {
                if let Some(t) = self.check_static_ref(expr, *ref_kind, *span) {
                    return t;
                }
                let writes = matches!(ref_kind, RefKind::Unique | RefKind::Alias);
                let slice_target = writes && matches!(**expr, HirExpr::Slice { .. });
                if slice_target {
                    // `&mut a[i..j]` is a write-borrow, not a read of `a[i..j]`.
                    self.skip_slice_init_check = true;
                }
                let ty = self.check_ref_expr(expr, *ref_kind, *span, true);
                self.skip_slice_init_check = false; // never let the flag leak past this expression
                if writes {
                    let root = match &**expr {
                        HirExpr::Ident(n, _) => Some(*n),
                        other => self.static_field_path(other).map(|(r, _)| r),
                    };
                    if let Some(r) = root {
                        self.note_written_behind_back(r);
                    }
                    self.optimistically_mark_mut_target_init(expr);
                    if slice_target {
                        self.optimistically_mark_mut_slice_init(expr);
                    }
                }
                ty
            }
            HirExpr::This { span } => {
                self.set_span(*span);
                self.check_this_expr(span)
            }
            HirExpr::ModuleAccess(access) => {
                self.set_span(access.span);
                self.check_module_access_expr(access)
            }
            HirExpr::Lambda { span, .. } => {
                self.set_span(*span);
                self.check_lambda(expr, None, false)
            }
            HirExpr::Index {
                object,
                index,
                span,
            } => {
                self.set_span(*span);

                self.check_index_expr(object, index)
            }
            HirExpr::ArrayLiteral { elements, span } => {
                self.set_span(*span);

                self.check_array_literal_expr(elements)
            }
            HirExpr::GenericIdent(name, type_args, span) => {
                self.set_span(*span);
                self.check_generic_ident_expr(expr, name, type_args)
            }
            HirExpr::Cast {
                expr,
                target_type,
                span,
            } => {
                self.set_span(*span);
                self.check_cast_expr(expr, target_type)
            }
            HirExpr::Char(_, _) => HirType::char(span),
            HirExpr::UnknownIntrinsic { span, name } => {
                self.recover(
                    Err(TypeErrorKind::Generic(format!("Unknown intrinsic: {}", name)).at(*span)),
                    (),
                );
                HirType::unknown(*span)
            }
            HirExpr::OrElse {
                value,
                else_body,
                span,
            } => {
                self.set_span(*span);
                self.check_or_else_expr(value, else_body)
            }
        }
    }

    fn collect_bound_names(c: &HirType<'a, 'bump>, out: &mut Vec<String>) {
        match c.kind {
            HirTypeKind::DynInterface { name: n, .. } | HirTypeKind::Struct { name: n, .. } => {
                out.push(n.to_string())
            }
            HirTypeKind::Dyn { bounds } => bounds
                .iter()
                .for_each(|b| Self::collect_bound_names(b, out)),
            _ => {}
        }
    }

    pub(crate) fn build_liveness(&self) -> Liveness {
        let n = self.cfg.point_count();
        let mut idx: FxHashMap<StrId, u32> = FxHashMap::default(); // separate from local_table on purpose
        for set in self.point_locals_used.values() {
            for &s in set {
                let k = idx.len() as u32;
                idx.entry(s).or_insert(k);
            }
        }
        let words = ((idx.len() + 63) / 64).max(1);
        let mut live = vec![0u64; n * words];
        for (p, set) in &self.point_locals_used {
            for s in set {
                let b = idx[s] as usize;
                live[p.0 as usize * words + b / 64] |= 1 << (b % 64);
            }
        }
        let mut changed = true;
        while changed {
            changed = false;
            for p in (0..n).rev() {
                // reverse creation order ≈ reverse program order
                let Some(succs) = self.cfg.successors.get(&PointId(p as u32)) else {
                    continue;
                };
                for s in succs {
                    for w in 0..words {
                        let add = live[s.0 as usize * words + w] & !live[p * words + w];
                        if add != 0 {
                            live[p * words + w] |= add;
                            changed = true;
                        }
                    }
                }
            }
        }
        Liveness { words, live, idx }
    }

    fn visibility_rank(v: Visibility) -> u8 {
        match v {
            Visibility::Private => 0,
            Visibility::Module => 1,
            Visibility::Internal => 2,
            Visibility::Public => 3,
        }
    }

    fn visibility_label(v: Visibility) -> &'static str {
        match v {
            Visibility::Private => "private",
            Visibility::Module => "module",
            Visibility::Internal => "internal",
            Visibility::Public => "public",
        }
    }

    fn collect_named_type_visibilities(
        &self,
        ty: &HirType<'a, 'bump>,
        out: &mut Vec<(String, Visibility)>,
    ) {
        use HirTypeKind as K;
        match ty.kind {
            K::Struct {
                name, type_args, ..
            } => {
                if let Some(def) = self.context.get_struct(name) {
                    out.push((def.unmangled_name.to_string(), def.visibility));
                }
                for a in type_args {
                    self.collect_named_type_visibilities(a, out);
                }
            }
            K::Enum {
                name, type_args, ..
            } => {
                if let Some(def) = self.context.get_enum(name) {
                    out.push((def.unmangled_name.to_string(), def.visibility));
                }
                for a in type_args {
                    self.collect_named_type_visibilities(a, out);
                }
            }
            K::DynInterface {
                name, type_args, ..
            } => {
                if let Some(def) = self.context.get_interface(name.as_str()) {
                    out.push((def.unmangled_name.to_string(), def.visibility));
                }
                for a in type_args {
                    self.collect_named_type_visibilities(a, out);
                }
            }
            K::Ref { inner, .. }
            | K::SafePointer { inner, .. }
            | K::UnsafePointer { inner, .. }
            | K::OwnedPointer { inner, .. }
            | K::Nullable(inner)
            | K::Slice(inner)
            | K::Array(inner, _)
            | K::Range { elem: inner, .. } => self.collect_named_type_visibilities(inner, out),
            K::Lambda {
                params,
                return_type,
            } => {
                for p in params {
                    self.collect_named_type_visibilities(p, out);
                }
                self.collect_named_type_visibilities(return_type, out);
            }
            K::Tuple(elems) => {
                for e in elems {
                    self.collect_named_type_visibilities(e, out);
                }
            }
            K::Dyn { bounds } => {
                for b in bounds {
                    self.collect_named_type_visibilities(b, out);
                }
            }
            _ => {}
        }
    }

    /// Call once per function declaration (free functions and methods).
    pub fn check_return_type_visibility(
        &mut self,
        func: &HirFunc<'a, 'bump>,
        span: SourceSpan<'a>,
    ) {
        let Some(ret) = func.return_type else { return };
        let func_vis = func.function_metadata.visibility;

        let mut found = Vec::new();
        self.collect_named_type_visibilities(&ret, &mut found);

        let mut seen: HashSet<String> = HashSet::default();
        for (type_name, type_vis) in found {
            if Self::visibility_rank(type_vis) < Self::visibility_rank(func_vis)
                && seen.insert(type_name.clone())
            {
                self.warnings.push(TypeWarning {
                    kind: TypeWarningKind::ReturnTypeLessVisible {
                        func_name: func.unmangled_name.to_string(),
                        func_visibility: Self::visibility_label(func_vis).into(),
                        type_name,
                        type_visibility: Self::visibility_label(type_vis).into(),
                    },
                    span,
                });
            }
        }
    }

    pub(crate) fn record_type_name_use(&mut self, span: SourceSpan<'a>, key: StrId) {
        if span.line == 0 {
            return;
        }
        let (name, unmangled, owner, ty) = if let Some(d) = self.context.get_struct(key) {
            (
                d.name,
                d.unmangled_name,
                self.context.struct_owner(key),
                HirType::struct_type(d.name, d.unmangled_name, &[], &[], span),
            )
        } else if let Some(d) = self.context.get_interface(key) {
            (
                d.name,
                d.unmangled_name,
                self.context.interface_owner(key),
                HirType::new(
                    HirTypeKind::DynInterface {
                        name: d.name,
                        unmangled_name: d.unmangled_name,
                        type_args: &[],
                    },
                    span,
                ),
            )
        } else if let Some(d) = self.context.get_enum(key) {
            (
                d.name,
                d.unmangled_name,
                self.context.enum_owner(key),
                HirType::new(
                    HirTypeKind::Enum {
                        name: d.name,
                        unmangled_name: d.unmangled_name,
                        variants: d.variants,
                        type_args: &[],
                    },
                    span,
                ),
            )
        } else {
            return;
        };
        if let Some(owner) = owner {
            self.record_item_occurrence_named(span, unmangled, name, ty, owner);
        }
    }

    pub fn forget_module(&mut self, module_idx: usize) {
        let g = std::sync::Arc::make_mut(&mut self.context.globals);

        let dead: Vec<StrId> = g
            .struct_owner_module
            .iter()
            .filter(|(_, m)| **m == module_idx)
            .map(|(k, _)| *k)
            .collect();
        for k in dead {
            g.structs.remove(&k);
            g.struct_owner_module.remove(&k);
        }

        let dead: Vec<StrId> = g
            .enum_owner_module
            .iter()
            .filter(|(_, m)| **m == module_idx)
            .map(|(k, _)| *k)
            .collect();
        for k in dead {
            g.enums.remove(&k);
            g.enum_owner_module.remove(&k);
        }

        let dead: Vec<StrId> = g
            .interface_owner_module
            .iter()
            .filter(|(_, m)| **m == module_idx)
            .map(|(k, _)| *k)
            .collect();
        for k in dead {
            g.interfaces.remove(&k);
            g.interface_owner_module.remove(&k);
        }

        g.module_functions.remove(&module_idx);
        for table in g.type_methods.values_mut() {
            table
                .methods
                .retain(|_, f| f.declaring_module_idx != module_idx);
        }
        self.module_consts.remove(&module_idx);
        self.module_statics.remove(&module_idx);
    }

    fn kill_facts_for_loop(&mut self, loop_stmt: &HirStmt<'a, 'bump>) {
        let mut v = AssignedVars::default();
        v.visit_stmt(loop_stmt);
        if v.names.is_empty() {
            return;
        }
        let names = v.names;
        self.borrow_checker
            .kill_facts_where(&|s| names.contains(&s));
        for &n in &names {
            self.kill_facts_about(n);
        }
    }
}

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use crate::closures;
use crate::initialization::{BindingMode, InitNode, ModuleImports};
use crate::move_state::MoveState;
use crate::naming::{str_id_to_string, type_to_string};
use crate::type_context::TypeContext;
use codex_dependency_graph::DepGraph;
use ir::analysis_context::CopyAnalysisCtx;
use ir::auto_imports::AutoImportRegistry;
use ir::borrow_checker::{BorrowChecker, LoanId, PlaceId, ReadTemplate, RefTemplate};
use ir::errors::type_error::{TypeCheckResult, TypeError, TypeErrorKind};
use ir::hir::{
    Hir, HirExpr, HirFunc, HirModule, HirParam, HirStmt, HirType, RefKind, StrId, ThisPassingKind,
};
use ir::ir_hasher::{FxHashMap, HashSet};
use ir::nll_cfg::{Cfg, CfgBuilder, PointId};
use ir::span::SourceSpan;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub const SLICE_PRIMITIVES: &[&str] = &["write_uninit", "write_uninit_all", "get_unchecked"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LocalSymbolId(pub u32);

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

pub type NonNullState = FxHashMap<StrId, HashSet<Vec<StrId>>>;

pub struct TypeChecker<'a, 'bump> {
    pub(crate) context: TypeContext<'a, 'bump>,
    pub(crate) errors: Vec<TypeError<'a>>,
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
    pub(crate) occurrences: Vec<(
        SourceSpan<'a>,
        StrId,
        HirType<'a, 'bump>,
        usize,
        SymbolId,
        bool,
    )>,
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
    pub(crate) init_state: FxHashMap<StrId, InitNode>,
    pub(crate) suppress_init_read: bool,
    pub(crate) local_ref_kind: FxHashMap<StrId, RefKind>,
    pub(crate) binding_mode_backfill: FxHashMap<usize, BindingMode>,
    pub(crate) non_null_state: NonNullState,
    pub(crate) in_place_context: bool,
    pub(crate) skip_slice_init_check: bool,
    pub(crate) closure_frames: std::cell::RefCell<Vec<closures::ClosureFrame<'bump>>>,
    pub(crate) closure_table: FxHashMap<usize, ir::hir::ClosureLowering<'a, 'bump>>,
    pub(crate) closure_loans: FxHashMap<usize, Vec<LoanId>>,
    pub(crate) next_closure_id: u32,
    pub(crate) fn_closure_constraints: FxHashMap<StrId, HirType<'a, 'bump>>,
    pub(crate) closure_pre_subs: FxHashMap<StrId, HirType<'a, 'bump>>,
    pub(crate) closure_generic_subs: FxHashMap<StrId, HirType<'a, 'bump>>,
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
            closure_frames: RefCell::new(Vec::default()),
            closure_table: FxHashMap::default(),
            closure_loans: FxHashMap::default(),
            next_closure_id: 0,
            fn_closure_constraints: FxHashMap::default(),
            closure_pre_subs: FxHashMap::default(),
            closure_generic_subs: FxHashMap::default(),
        }
    }

    pub fn scope_end_init_snapshot(&self, name: StrId) -> Option<&InitNode> {
        self.init_state.get(&name)
    }

    pub fn uninit_ty(&self, expr: &HirExpr<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
        self.uninit_backfill.get(&Self::expr_key(expr)).copied()
    }

    pub fn record_item_occurrence(
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

    pub fn record_method_occurrence(
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

    #[inline]
    pub fn set_span(&mut self, span: SourceSpan<'a>) {
        self.current_span = span;
    }

    pub fn slice_field_owned(ty: &HirType<'a, 'bump>) -> Option<bool> {
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
            .map(|p| {
                p.iter()
                    .map(|s| StrId(self.context.string_pool.intern(s)))
                    .collect()
            })
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
                            .map(|p| p.to_string())
                            .unwrap_or_default();
                        let new_pkg = self
                            .context
                            .dep_graph
                            .borrow()
                            .get_module_package(target_module)
                            .map(|p| p.to_string())
                            .unwrap_or_default();
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

    pub fn check_function(&mut self, func: &HirFunc<'a, 'bump>) {
        let mut func_context = self.context.create_child_scope();

        self.fn_closure_constraints.clear();
        if let Some(gs) = func.generics {
            for g in gs.iter() {
                if let Some(c) = g
                    .constraints
                    .iter()
                    .find(|c| matches!(c, HirType::Lambda { .. }))
                {
                    self.fn_closure_constraints.insert(g.name, *c);
                }
            }
        }

        self.borrow_checker = BorrowChecker::new();
        self.borrow_checker.begin_scope();

        if let Some(params) = func.params {
            for param in params.iter() {
                match param {
                    HirParam::Normal {
                        name, param_type, ..
                    } => {
                        let param_name = str_id_to_string(*name);
                        let symbol_id = self.mint_symbol_id();
                        if func_context.get_variable(&param_name).is_some() {
                            self.record(TypeErrorKind::VariableAlreadyExists {
                                var_name: param_name.clone(),
                            });
                        }
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

                        let self_ty = self.this_type_for_func(func);
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

    pub fn this_type_for_func(&self, func: &HirFunc<'a, 'bump>) -> HirType<'a, 'bump> {
        let Some(target) = func.impl_target else {
            return HirType::This;
        };
        let target_str = str_id_to_string(target);

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

    pub fn stmt_key(stmt: &HirStmt<'a, 'bump>) -> usize {
        stmt as *const HirStmt<'a, 'bump> as usize
    }

    pub fn set_point(&mut self, stmt: &HirStmt<'a, 'bump>) {
        if let Some(&point) = self.stmt_points.get(&Self::stmt_key(stmt)) {
            self.current_point = point;
        }
    }

    pub fn check_stmt(&mut self, stmt: &HirStmt<'a, 'bump>) -> Option<HirType<'a, 'bump>> {
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
                self.set_span(*span);
                self.check_let_stmt(name, ty, value, mutable, else_block, span)
            }
            HirStmt::Return(expr, span) => {
                self.set_span(*span);
                self.check_return_stmt(expr)
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
            HirStmt::While { cond, body } => self.check_while_stmt(cond, body),

            HirStmt::For {
                init,
                condition,
                increment,
                body,
            } => self.check_for_stmt(init, condition, increment, body),
            HirStmt::Block { body, span } => {
                self.set_span(*span);
                self.check_block_stmt(body)
            }
            HirStmt::Break(expr, span) => {
                self.set_span(*span);
                self.check_break_stmt(expr)
            }
            HirStmt::Continue(span) => {
                self.set_span(*span);
                if !self.context.in_loop {
                    self.record(TypeErrorKind::ContinueOutsideLoop);
                }
                Some(HirType::Never)
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

    pub fn leaf_span(expr: &HirExpr<'a, 'bump>) -> Option<SourceSpan<'a>> {
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

    pub fn check_expr_expected(
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
                    type_to_string(expected)
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
            } => self.check_enum_init_expr(
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

            HirExpr::Lambda { .. } => match expected {
                HirType::Lambda { .. } => self.check_lambda(expr, Some(*expected), false),
                _ => self.check_lambda(expr, None, false),
            },

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
                let var_name = str_id_to_string(*name);
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
                                type_to_string(&object_ty)
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
                            type_to_string(&object_ty)
                        )));
                        HirType::Unknown
                    }
                }
            }
            _ => self.check_expr(expr),
        }
    }

    pub fn check_expr(&mut self, expr: &HirExpr<'a, 'bump>) -> HirType<'a, 'bump> {
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
                HirType::Tuple(self.context.bump.alloc_slice_copy(types.as_slice()))
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
                self.check_stmt(if_stmt).unwrap_or(HirType::Void)
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
            } => self.check_call_expr(expr, callee, args, span, type_args),
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
                self.check_struct_init_expr(name, args, type_args)
            }
            HirExpr::InterfaceCall {
                callee,
                interface,
                args,
                span,
            } => {
                self.set_span(*span);
                self.check_interface_call_expr(callee, interface, args)
            }

            HirExpr::Assignment {
                target,
                op,
                value,
                span,
            } => {
                self.set_span(*span);
                self.check_assignment_expr(target, op, value)
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
            } => self.check_enum_init_expr(expr, enum_name, variant, args, type_args, *span, None),
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
                let ty = self.check_ref_expr(expr, *ref_kind, *span, true);
                if matches!(ref_kind, RefKind::Unique | RefKind::Alias) {
                    self.optimistically_mark_mut_target_init(expr);
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
            HirExpr::GenericIdent(..) => todo!(),
            HirExpr::Cast {
                expr,
                target_type,
                span,
            } => {
                self.set_span(*span);
                self.check_cast_expr(expr, target_type)
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
}

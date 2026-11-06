use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use std::time::{Duration, Instant};

use codex_dependency_graph::dep_graph::{AstModule, DepGraph};
use engraver_assembly_emit::backend::Backend;
use ir::analysis_context::CopyAnalysisCtx;
use ir::ast::Stmt;
use ir::auto_imports::AutoImportRegistry;
use ir::errors::reporter::ErrorReporter;
use ir::errors::type_error::{TypeError, TypeErrorKind};
use ir::hir::{Hir, HirEnum, HirModule, HirStruct, StrId};
use ir::ir_hasher::{FxHashMap, HashMap, HashSet};
use ir::registry::global_registry::GlobalRegistry;
use scribe_parser::hir_lowerer::HirLowerer;
use scribe_parser::hir_lowerer::dce::{DceConfig, DeadCodeEliminator};
use scribe_parser::hir_lowerer::lambda_hoisting::LambdaHoister;
use scribe_parser::hir_lowerer::monomorphization::Monomorphizer;
use sentinel_typechecker::TypeChecker;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub mod file_handling;
pub mod file_loader;
#[cfg(target_os = "linux")]
pub mod io_uring_file_loader;
#[cfg(target_os = "windows")]
pub mod iocp_file_loader;
pub mod link;
pub mod main_structs;
pub mod std_file_loader;

use crate::file_loader::FileLoader;
use crate::main_structs::{CompilerError as BuildError, ModuleWithArena};

#[derive(Default)]
struct CompilerTimings {
    frontend: Duration,
    typecheck_borrowck: Duration,
    monomorphization: Duration,
    codegen: Duration,
    linking: Duration,
}

impl CompilerTimings {
    fn report_check(&self) {
        let stages: [(&str, Duration); 2] = [
            ("Frontend", self.frontend),
            ("Type & borrow checking", self.typecheck_borrowck),
        ];
        let total = self.frontend + self.typecheck_borrowck;
        Self::display_report(&stages, total);
    }

    fn report_with_linking(&self) {
        let stages: [(&str, Duration); 5] = [
            ("Frontend", self.frontend),
            ("Type & borrow checking", self.typecheck_borrowck),
            ("Monomorphization", self.monomorphization),
            ("Codegen", self.codegen),
            ("Linking", self.linking),
        ];

        let total = self.frontend
            + self.typecheck_borrowck
            + self.monomorphization
            + self.codegen
            + self.linking;

        Self::display_report(&stages, total);
    }

    fn report(&self) {
        let stages: [(&str, Duration); 4] = [
            ("Frontend", self.frontend),
            ("Type & borrow checking", self.typecheck_borrowck),
            ("Monomorphization", self.monomorphization),
            ("Codegen", self.codegen),
        ];

        let total = self.frontend + self.typecheck_borrowck + self.monomorphization + self.codegen;

        Self::display_report(&stages, total);
    }

    fn display_report(stages: &[(&str, Duration)], total: Duration) {
        let total_secs = total.as_secs_f64();

        let rows: Vec<String> = stages
            .iter()
            .map(|(name, elapsed)| {
                let ms = elapsed.as_secs_f64() * 1_000.0;
                let percentage = if total_secs > 0.0 {
                    elapsed.as_secs_f64() / total_secs * 100.0
                } else {
                    0.0
                };

                format!("{name:<28} {ms:>10.3} ms  {percentage:>5.1}%")
            })
            .collect();

        let total_row = format!("{:<28} {:>10.3} ms", "Total measured", total_secs * 1_000.0);

        let title = "Zeta Compiler Timings";
        let content_width = rows
            .iter()
            .map(String::len)
            .chain([total_row.len(), title.len()])
            .max()
            .unwrap_or(0)
            + 4;

        let border = "─".repeat(content_width);

        eprintln!();
        eprintln!("┌{border}┐");
        eprintln!("│{:^content_width$}│", title);
        eprintln!("├{border}┤");

        for row in &rows {
            eprintln!("│  {row:<width$}  │", width = content_width - 4);
        }

        eprintln!("├{border}┤");
        eprintln!("│  {total_row:<width$}  │", width = content_width - 4);
        eprintln!("└{border}┘");
        eprintln!();
    }
}

pub struct Compiler<'a, 'bump> {
    pool: Arc<StringPool>,
    registry: GlobalRegistry<'a, 'bump>,
    auto_imports: Rc<RefCell<AutoImportRegistry>>,

    dep_graph: &'a RefCell<DepGraph<'a>>,
    #[allow(unused)] // Avoids a UB
    dep_graph_storage: Box<RefCell<DepGraph<'a>>>,
    type_checker: Rc<RefCell<TypeChecker<'a, 'bump>>>,
    cpy_ctx: Rc<RefCell<CopyAnalysisCtx<'a, 'bump>>>,

    module_ids: HashMap<StrId, usize>,
    next_module_idx: usize,
    stdlib_module_ids: HashSet<usize>,
    modules: HashMap<usize, ModuleWithArena<'a, 'bump>>,
    hir_modules: HashMap<usize, HirModule<'a, 'bump>>, // pre-monomorphization
    codegen_hir_modules: HashMap<usize, HirModule<'a, 'bump>>, // post-monomorphization
    loaded_sources: HashMap<String, String>,
    #[allow(unused)] // Avoids a UB
    lowerer_bump: Box<GrowableBump<'bump>>,
    source_roots: Vec<PathBuf>,
    env_structs: FxHashMap<StrId, StrId>,
    /// Modules that passed type checking, in the order they must be monomorphized.
    pending_mono: Vec<usize>,
    /// Lowerers kept until deferred monomorphization (mono runs against their ctx).
    deferred_lowerers: Vec<HirLowerer<'a, 'bump>>,
    /// module_idx -> index into `deferred_lowerers`.
    deferred_lowerer_slot: FxHashMap<usize, usize>,
    dce_enabled: bool,
    timings: CompilerTimings,
    retired_modules: Vec<ModuleWithArena<'a, 'bump>>,
}

impl<'a, 'bump> Compiler<'a, 'bump>
where
    'bump: 'a,
    'a: 'bump,
{
    pub fn new() -> Result<Self, BuildError<'a>> {
        let pool = Arc::new(StringPool::new().map_err(|_| BuildError::FailedToAllocateStringPool)?);
        let registry = GlobalRegistry::new();

        let dep_graph_storage = Box::new(RefCell::new(DepGraph::new()));
        // SAFETY: Box's heap allocation is stable across Compiler moves.
        // dep_graph_storage's contents are only ever mutated through
        // `.borrow_mut()` on this erased reference or on the boxed value
        // directly, never replaced wholesale. RefCell's runtime borrow
        // tracking is what makes this genuinely sound (not just
        // borrow-checker-satisfied): any illegal aliasing that would occur
        // panics at the .borrow()/.borrow_mut() call site instead of
        // silently producing UB
        let dep_graph: &'a RefCell<DepGraph> =
            unsafe { &*(dep_graph_storage.as_ref() as *const RefCell<DepGraph>) };

        let lowerer_bump = Box::new(GrowableBump::new(4096, 8));
        let lowerer_bump_ref: &'bump GrowableBump<'bump> =
            unsafe { &*(lowerer_bump.as_ref() as *const GrowableBump<'bump>) };

        let auto_imports = Rc::new(RefCell::new(AutoImportRegistry::new()));

        let cpy_ctx = Rc::new(RefCell::new(CopyAnalysisCtx::new(&[], registry.clone())));
        let type_checker = Rc::new(RefCell::new(TypeChecker::new(
            dep_graph,
            lowerer_bump_ref,
            cpy_ctx.clone(),
            pool.clone(),
            auto_imports.clone(),
        )));

        Ok(Self {
            pool,
            registry,
            auto_imports,
            dep_graph,
            dep_graph_storage,
            type_checker,
            cpy_ctx,
            lowerer_bump,
            module_ids: HashMap::default(),
            next_module_idx: 0,
            stdlib_module_ids: HashSet::default(),
            modules: HashMap::default(),
            hir_modules: HashMap::default(),
            codegen_hir_modules: HashMap::default(),
            loaded_sources: HashMap::default(),
            source_roots: Vec::default(),
            env_structs: FxHashMap::default(),
            pending_mono: Vec::new(),
            deferred_lowerers: Vec::new(),
            deferred_lowerer_slot: FxHashMap::default(),
            dce_enabled: true,
            timings: CompilerTimings::default(),
            retired_modules: Vec::default(),
        })
    }

    pub fn report_check_timings(&self) {
        self.timings.report_check();
    }

    pub fn set_dead_code_elimination(&mut self, enabled: bool) {
        self.dce_enabled = enabled;
    }

    pub fn dep_graph(&self) -> &'a RefCell<DepGraph<'a>> {
        self.dep_graph
    }

    pub fn path_for_module(&self, module_idx: usize) -> Option<&Path> {
        self.modules.get(&module_idx).map(|m| m.path.as_path())
    }

    pub fn module_idx_for_path(&self, path: &Path) -> Option<usize> {
        self.module_ids.get(&self.canonical_key(path)).copied()
    }

    fn register_implicit_imports(&self, module_idx: usize) {
        let mut g = self.dep_graph.borrow_mut();
        if self.stdlib_module_ids.contains(&module_idx) {
            for &existing in self.module_ids.values() {
                if existing != module_idx && !self.stdlib_module_ids.contains(&existing) {
                    g.register_import(existing, module_idx);
                }
            }
        } else {
            for &std_idx in &self.stdlib_module_ids {
                g.register_import(module_idx, std_idx);
            }
        }
    }

    pub fn type_checker(&self) -> Rc<RefCell<TypeChecker<'a, 'bump>>> {
        self.type_checker.clone()
    }

    pub fn dep_graph_ref(&self) -> &'a RefCell<DepGraph<'a>> {
        self.dep_graph
    }

    pub fn source_text(&self, module_idx: usize) -> Option<String> {
        Some(self.modules.get(&module_idx)?.source.to_string())
    }

    pub fn load_directory<L: FileLoader>(
        &mut self,
        loader: &L,
        dir: &Path,
        is_stdlib: bool,
    ) -> Result<ErrorReporter<'a>, BuildError<'a>> {
        let frontend_start = Instant::now();
        self.source_roots.push(dir.to_path_buf());

        let files = crate::file_handling::collect_zeta_files(dir)?;
        let sources = loader
            .load_files(&files)
            .map_err(|e| BuildError::FailedToReadFile(dir.to_path_buf(), e))?;

        let mut reporter = ErrorReporter::new();
        let mut batch: Vec<(usize, StrId, ModuleWithArena<'a, 'bump>)> = Vec::new();

        for file in sources {
            let name = self.canonical_key(&file.path);
            let module_idx = self.module_idx_for(name);

            let parsed = crate::file_handling::parse_single_file_from_source(
                self.pool.clone(),
                file.path,
                file.source,
            )?;

            self.loaded_sources
                .insert(name.to_string(), parsed.source.to_string());

            for perr in &parsed.parse_result.diagnostics.errors {
                reporter.add_parser_error(perr.clone());
            }

            if is_stdlib {
                self.stdlib_module_ids.insert(module_idx);
            }

            let ast_module = codex_dependency_graph::dep_graph::AstModule {
                name,
                path: parsed.path.clone(),
                stmts: parsed.parse_result.statements.as_slice(),
            };
            self.dep_graph.borrow_mut().register_module_structure(
                module_idx,
                &ast_module,
                &self.pool,
            );

            batch.push((module_idx, name, parsed));
        }

        for (module_idx, name, parsed) in &batch {
            let ast_module = codex_dependency_graph::dep_graph::AstModule {
                name: *name,
                path: parsed.path.clone(),
                stmts: parsed.parse_result.statements.as_slice(),
            };
            self.dep_graph.borrow_mut().extract_edges_for_module(
                *module_idx,
                &ast_module,
                &self.pool,
            );
            self.register_implicit_imports(*module_idx);
        }

        if is_stdlib {
            for (module_idx, _, _) in &batch {
                for &existing_idx in self.module_ids.values() {
                    if existing_idx != *module_idx
                        && !self.stdlib_module_ids.contains(&existing_idx)
                    {
                        self.dep_graph
                            .borrow_mut()
                            .register_import(existing_idx, *module_idx);
                    }
                }
            }
        } else {
            for (module_idx, _, _) in &batch {
                for &stdlib_idx in &self.stdlib_module_ids {
                    self.dep_graph
                        .borrow_mut()
                        .register_import(*module_idx, stdlib_idx);
                }
            }
        }

        let mut pending: FxHashMap<usize, (StrId, ModuleWithArena<'a, 'bump>)> = batch
            .into_iter()
            .map(|(idx, name, parsed)| (idx, (name, parsed)))
            .collect();

        let order = self.dep_graph.borrow().get_module_compilation_order();

        self.timings.frontend += frontend_start.elapsed();

        #[cfg(debug_assertions)]
        {
            use ir::ir_hasher::FxHashMap;
            let names: FxHashMap<usize, String> = self
                .module_ids
                .iter()
                .map(|(&name, &idx)| (idx, name.to_string()))
                .collect();
            self.dep_graph.borrow().debug_dump_sccs();
            self.dep_graph.borrow().debug_dump_module_order(&names);
        }

        for (path, source) in &self.loaded_sources {
            reporter.add_source_file(path.clone(), source.clone());
        }
        self.compile_pending_grouped(&order, &mut pending, &mut reporter);

        for (module_idx, (_name, parsed)) in pending {
            reporter.merge(self.lower_and_check_module(module_idx, parsed));
        }

        let mismatches = self
            .dep_graph
            .borrow()
            .check_package_paths(&[dir.to_path_buf()]);
        for m in &mismatches {
            eprintln!("error: {}", m);
        }

        let unresolved = self.dep_graph.borrow().unresolved_import_messages();
        for msg in &unresolved {
            eprintln!("error: {}", msg);
        }

        if !mismatches.is_empty() || !unresolved.is_empty() {
            return Err(BuildError::InvalidModuleStructure {
                package_mismatches: mismatches.len(),
                unresolved_imports: unresolved.len(),
            });
        }

        Ok(reporter)
    }

    fn make_lowerer(&self, parsed: &ModuleWithArena<'a, 'bump>) -> HirLowerer<'a, 'bump> {
        let module_auto_imports = self.auto_imports.clone();
        HirLowerer::new(
            self.pool.clone(),
            parsed.bump_ref(),
            self.dep_graph,
            self.registry.clone(),
            module_auto_imports,
        )
    }

    fn lower_module_bodies_phase(
        &mut self,
        module_idx: usize,
        mut lowerer: HirLowerer<'a, 'bump>,
        parsed: ModuleWithArena<'a, 'bump>,
        reporter: &mut ErrorReporter<'a>,
    ) -> Option<HirLowerer<'a, 'bump>> {
        let hir = lowerer.lower_module_bodies(&parsed.parse_result.statements, module_idx);

        for err in lowerer.lowering_errors().iter() {
            reporter.add_type_error(TypeError {
                kind: TypeErrorKind::Generic(err.0.clone()),
                span: err.1.clone(),
            });
        }

        self.hir_modules.insert(module_idx, hir);
        self.modules.insert(module_idx, parsed);

        if reporter.has_errors() {
            None
        } else {
            Some(lowerer)
        }
    }

    fn finalize_monomorphization(&mut self) {
        let mut emitted: HashSet<StrId> = HashSet::default();
        let mut by_module: FxHashMap<usize, Vec<Hir<'a, 'bump>>> = FxHashMap::default();

        loop {
            let before = self.registry.instantiated_functions.borrow().len()
                + self.registry.instantiated_structs.borrow().len()
                + self.registry.instantiated_enums.borrow().len();

            self.force_instantiate_drops();
            self.force_instantiate_allocator_frees();

            let after = self.registry.instantiated_functions.borrow().len()
                + self.registry.instantiated_structs.borrow().len()
                + self.registry.instantiated_enums.borrow().len();
            if before == after {
                break;
            }
        }

        for new_name in self.registry.instantiated_functions.borrow().values() {
            if !emitted.insert(*new_name) {
                continue;
            }

            if let Some(func) = self.registry.functions.borrow().get(new_name) {
                by_module
                    .entry(func.declaring_module_idx)
                    .or_default()
                    .push(Hir::Func(
                        self.lowerer_bump.alloc_value_immutable(func.clone()),
                    ));
            }
        }

        for new_struct_name in self.registry.instantiated_structs.borrow().values() {
            if !emitted.insert(*new_struct_name) {
                continue;
            }
            let Some(new_struct) = self.first_ctx_structs_lookup(new_struct_name) else {
                continue;
            };
            let Some((origin_name, _)) = self
                .registry
                .instantiated_struct_origins
                .borrow()
                .get(new_struct_name)
                .cloned()
            else {
                continue;
            };
            let Some(&owner) = self.registry.struct_owner_module.borrow().get(&origin_name) else {
                continue;
            };
            by_module.entry(owner).or_default().push(Hir::Struct(
                self.lowerer_bump.alloc_value_immutable(new_struct),
            ));
        }

        for new_enum_name in self.registry.instantiated_enums.borrow().values() {
            if !emitted.insert(*new_enum_name) {
                continue;
            }
            let Some(new_enum) = self.first_ctx_enums_lookup(new_enum_name) else {
                continue;
            };
            let Some((origin_name, _)) = self
                .registry
                .instantiated_enum_origins
                .borrow()
                .get(new_enum_name)
                .cloned()
            else {
                continue;
            };
            let Some(&owner) = self.registry.enum_owner_module.borrow().get(&origin_name) else {
                continue;
            };
            by_module
                .entry(owner)
                .or_default()
                .push(Hir::Enum(self.lowerer_bump.alloc_value_immutable(new_enum)));
        }

        for (module_idx, extra_items) in by_module {
            let Some(existing) = self.codegen_hir_modules.get(&module_idx) else {
                continue;
            };
            let mut merged: Vec<Hir<'a, 'bump>> = existing.items.to_vec();
            merged.extend(extra_items);
            let bump = self
                .modules
                .get(&module_idx)
                .map(|m| &m.bump)
                .expect("Expected module idx to exist");
            let new_module = HirModule {
                name: existing.name,
                imports: existing.imports,
                items: bump.alloc_slice(&merged),
            };
            self.codegen_hir_modules.insert(module_idx, new_module);
        }
    }

    fn monomorphize_module(
        &mut self,
        module_idx: usize,
        checked_hir: HirModule<'a, 'bump>,
        lowerer: &mut HirLowerer<'a, 'bump>,
    ) {
        let hoister = LambdaHoister::new(
            lowerer.ctx.bump,
            self.pool.clone(),
            checked_hir.name,
            self.type_checker.borrow().closure_table().clone(),
            module_idx,
        );
        let (hoisted_module, env_to_fn) = hoister.run(checked_hir);

        self.env_structs
            .extend(env_to_fn.iter().map(|(&k, &v)| (k, v)));

        let mut monomorphizer = Monomorphizer::new(
            self.pool.clone(),
            lowerer.ctx.bump,
            self.registry.functions.clone(),
            &mut lowerer.ctx,
            self.registry.instantiated_functions.clone(),
            self.registry.instantiated_structs.clone(),
            self.registry.instantiated_struct_origins.clone(),
            self.registry.instantiated_enums.clone(),
            self.registry.instantiated_enum_origins.clone(),
            env_to_fn,
        );
        let monomorphized_module = monomorphizer.run(hoisted_module);

        self.codegen_hir_modules
            .insert(module_idx, monomorphized_module);
    }

    fn compile_pending_grouped(
        &mut self,
        order: &[usize],
        pending: &mut FxHashMap<usize, (StrId, ModuleWithArena<'a, 'bump>)>,
        reporter: &mut ErrorReporter<'a>,
    ) {
        if pending.is_empty() {
            return;
        }

        let mut seen: HashSet<usize> = HashSet::default();
        let mut groups: Vec<Vec<usize>> = Vec::new();
        for &module_idx in order {
            if !seen.insert(module_idx) {
                continue;
            }
            let scc = self.dep_graph.borrow().modules_in_same_scc(module_idx);
            for &m in &scc {
                seen.insert(m);
            }
            groups.push(scc);
        }

        let frontend_start = Instant::now();

        let bump = self.scratch_bump_ref();
        let mut lowerer = HirLowerer::new(
            self.pool.clone(),
            bump, // was &bump
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );

        let lowerer_slot = self.deferred_lowerers.len();
        let mut queued_any = false;

        let module_stmts: FxHashMap<usize, Vec<Stmt<'a, 'bump>>> = pending
            .iter()
            .map(|(&idx, (_name, parsed))| (idx, parsed.parse_result.statements.clone()))
            .collect();

        let mut lowered = lowerer.lower_all_modules(&module_stmts, order);

        self.timings.frontend += frontend_start.elapsed();

        let checking_start = Instant::now();

        self.type_checker
            .borrow_mut()
            .check_attributes(&lowerer.ctx.attrs.borrow());

        self.timings.typecheck_borrowck += checking_start.elapsed();

        for err in lowerer.lowering_errors().iter() {
            reporter.add_type_error(TypeError {
                kind: TypeErrorKind::Generic(err.0.clone()),
                span: err.1.clone(),
            });
        }

        for (module_idx, (_name, parsed)) in pending.drain() {
            if let Some(hir) = lowered.remove(&module_idx) {
                self.hir_modules.insert(module_idx, hir);
            }
            self.modules.insert(module_idx, parsed);
        }

        let checking_start = Instant::now();
        for group in &groups {
            let ready: Vec<usize> = group
                .iter()
                .copied()
                .filter(|m| self.hir_modules.contains_key(m))
                .collect();
            if ready.is_empty() {
                continue;
            }

            {
                let mut checker = self.type_checker.borrow_mut();
                for &module_idx in &ready {
                    if let Some(hir) = self.hir_modules.get(&module_idx) {
                        checker.register_module(hir, module_idx);
                    }
                }
            }

            let updated: Vec<(usize, &HirModule<'a, 'bump>)> = ready
                .iter()
                .filter_map(|&m| self.hir_modules.get(&m).map(|h| (m, h)))
                .collect();
            self.cpy_ctx.borrow_mut().recompute(&updated);

            {
                let mut checker = self.type_checker.borrow_mut();
                for &module_idx in &ready {
                    if let Some(hir) = self.hir_modules.get(&module_idx) {
                        checker.check_module_body(hir, module_idx);
                    }
                }
                for err in checker.take_errors() {
                    reporter.add_type_error(err);
                }
            }

            if reporter.has_errors() {
                continue;
            }

            for &module_idx in &ready {
                self.pending_mono.push(module_idx);
                self.deferred_lowerer_slot.insert(module_idx, lowerer_slot);
                queued_any = true;
            }
        }
        self.timings.typecheck_borrowck += checking_start.elapsed();

        if queued_any {
            self.deferred_lowerers.push(lowerer);
        }
    }

    fn lower_and_check_module(
        &mut self,
        module_idx: usize,
        parsed: ModuleWithArena<'a, 'bump>,
    ) -> ErrorReporter<'a> {
        let mut reporter = self.make_reporter();
        let frontend_start = Instant::now();

        let mut lowerer = self.make_lowerer(&parsed);
        lowerer.lower_module_prototypes(&parsed.parse_result.statements, module_idx);

        let lowered = self.lower_module_bodies_phase(module_idx, lowerer, parsed, &mut reporter);

        self.timings.frontend += frontend_start.elapsed();

        let Some(lowerer) = lowered else {
            return reporter;
        };

        let checking_start = Instant::now();
        let invalidation_set = self
            .dep_graph
            .borrow()
            .reverse_deps_transitive_modules(module_idx);

        {
            let mut checker = self.type_checker.borrow_mut();
            for &m in &invalidation_set {
                checker.forget_module(m);
            }

            for &m in &invalidation_set {
                if let Some(hir) = self.hir_modules.get(&m) {
                    checker.register_module(hir, m);
                }
            }
        }

        let updated: Vec<(usize, &HirModule<'a, 'bump>)> = invalidation_set
            .iter()
            .filter_map(|&m| self.hir_modules.get(&m).map(|h| (m, h)))
            .collect();
        self.cpy_ctx.borrow_mut().recompute(&updated);

        {
            let mut checker = self.type_checker.borrow_mut();
            for &m in &invalidation_set {
                if let Some(hir) = self.hir_modules.get(&m) {
                    checker.check_module_body(hir, m);
                }
            }
            for err in checker.take_errors() {
                reporter.add_type_error(err);
            }
        }

        self.timings.typecheck_borrowck += checking_start.elapsed();

        if reporter.has_errors() {
            return reporter;
        }

        let slot = self.deferred_lowerers.len();
        self.deferred_lowerers.push(lowerer);
        self.deferred_lowerer_slot.insert(module_idx, slot);
        self.pending_mono.push(module_idx);
        reporter
    }

    /// DCE over everything that passed type checking, then hoist + monomorphize.
    fn run_deferred_monomorphization(&mut self) {
        let order: Vec<usize> = std::mem::take(&mut self.pending_mono)
            .into_iter()
            .filter(|idx| self.hir_modules.contains_key(idx))
            .collect();
        if order.is_empty() {
            return;
        }
        let mut lowerers = std::mem::take(&mut self.deferred_lowerers);
        let slots = std::mem::take(&mut self.deferred_lowerer_slot);

        let checked: Vec<HirModule<'a, 'bump>> =
            order.iter().map(|idx| self.hir_modules[idx]).collect();
        let pruned = self.eliminate_dead_code(checked);

        for (module_idx, hir) in order.into_iter().zip(pruned) {
            let slot = slots[&module_idx];
            self.monomorphize_module(module_idx, hir, &mut lowerers[slot]);
        }
    }

    fn eliminate_dead_code(&self, modules: Vec<HirModule<'a, 'bump>>) -> Vec<HirModule<'a, 'bump>> {
        if !self.dce_enabled {
            return modules;
        }
        let cfg = DceConfig::new(&self.pool);

        // Library / object-only builds have no entry point: keep everything.
        let has_entry = modules.iter().any(|m| {
            m.items.iter().any(|it| {
                matches!(it, Hir::Func(f)
                    if cfg.entry_points.iter().any(|e| *e == f.name || *e == f.unmangled_name))
            })
        });
        if !has_entry {
            return modules;
        }

        let mut dce = DeadCodeEliminator::new(&cfg, self.scratch_bump_ref());
        let out = dce.run(&modules);
        #[cfg(debug_assertions)]
        if std::env::var_os("ZETA_DCE_LIST").is_some() {
            eprintln!("[zeta-debug] dce: {:?}", dce.stats);
        }
        out
    }

    pub fn emit(
        &mut self,
        out_dir: &Path,
        optimize: bool,
        verbose: bool,
        emit_obj: bool,
    ) -> Result<PathBuf, BuildError<'a>> {
        let monomorphization_start = Instant::now();

        self.run_deferred_monomorphization();
        self.finalize_monomorphization();

        self.timings.monomorphization += monomorphization_start.elapsed();
        let missing: Vec<PathBuf> = self
            .hir_modules
            .keys()
            .filter(|idx| !self.codegen_hir_modules.contains_key(idx))
            .filter_map(|idx| self.path_for_module(*idx).map(|p| p.to_path_buf()))
            .collect();
        if !missing.is_empty() {
            return Err(BuildError::CompilationAborted {
                reason: format!(
                    "{} module(s) failed type checking and were not compiled:\n  {}",
                    missing.len(),
                    missing
                        .iter()
                        .map(|p| p.display().to_string())
                        .collect::<Vec<_>>()
                        .join("\n  "),
                ),
            });
        }

        let compilation_order = self.dep_graph.borrow().get_module_compilation_order();

        let ordered_hir: Vec<HirModule<'a, 'bump>> = (0..self.next_module_idx)
            .filter_map(|i| self.codegen_hir_modules.get(&i).copied())
            .collect();

        let (codegen_start, backend) = crate::file_handling::emit_all(
            &ordered_hir,
            &compilation_order,
            self.pool.clone(),
            Rc::new(crate::file_handling::collect_extern_c_names(&ordered_hir)),
            self.dep_graph,
            self.registry.clone(),
            optimize,
            verbose,
        );

        let out_obj = match backend.finish(&out_dir.to_path_buf()) {
            Ok(path) => path,
            Err(e) => {
                self.timings.codegen += codegen_start.elapsed();
                self.timings.report();

                return Err(BuildError::FinishError(Box::new(e)));
            }
        };

        self.timings.codegen += codegen_start.elapsed();

        if emit_obj {
            self.timings.report();

            Ok(out_obj)
        } else {
            let linking_start = Instant::now();
            let program_path = out_dir.join("program");

            let link_result = crate::link::link(
                &[out_obj.to_str().unwrap()],
                program_path.to_str().unwrap(),
                true,
            );

            self.timings.linking += linking_start.elapsed();
            self.timings.report_with_linking();

            link_result.map(|_| program_path)
        }
    }

    pub fn module_with_arena(&self, module_idx: usize) -> Option<&ModuleWithArena<'a, 'bump>> {
        self.modules.get(&module_idx)
    }

    fn module_idx_for(&mut self, canonical_name: StrId) -> usize {
        if let Some(&idx) = self.module_ids.get(&canonical_name) {
            return idx;
        }
        let idx = self.next_module_idx;
        self.next_module_idx += 1;
        self.module_ids.insert(canonical_name, idx);
        idx
    }

    pub fn open_module(&mut self, path: &Path, source: String) -> ErrorReporter<'a> {
        self.update_module(path, source)
    }

    pub fn update_module(&mut self, path: &Path, source: String) -> ErrorReporter<'a> {
        let canonical_name = self.canonical_key(path);
        let module_idx = self.module_idx_for(canonical_name);

        if let Some(m) = self.modules.get(&module_idx) {
            if &m.source == &source && self.hir_modules.contains_key(&module_idx) {
                return self.make_reporter();
            }
        }

        let parsed = match crate::file_handling::parse_single_file_from_source(
            self.pool.clone(),
            PathBuf::from(canonical_name.as_str()),
            source,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: {}", e);
                return self.make_reporter();
            }
        };

        self.loaded_sources
            .insert(canonical_name.to_string(), parsed.source.to_string());
        let mut reporter = self.make_reporter();

        for perr in &parsed.parse_result.diagnostics.errors {
            reporter.add_parser_error(perr.clone());
        }
        if parsed.parse_result.diagnostics.has_errors() {
            return reporter;
        }

        let is_new = !self.modules.contains_key(&module_idx);
        let importers = self.dep_graph.borrow().get_module_importers(module_idx);

        {
            let ast_module = AstModule {
                name: canonical_name,
                path: parsed.path.clone(),
                stmts: parsed.parse_result.statements.as_slice(),
            };
            let mut g = self.dep_graph.borrow_mut();
            if is_new {
                g.register_module_structure(module_idx, &ast_module, &self.pool);
            } else {
                g.update_module_items(module_idx, &ast_module, &self.pool);
            }
            g.extract_edges_for_module(module_idx, &ast_module, &self.pool);
        }
        self.register_implicit_imports(module_idx);

        for imp_idx in importers {
            if let Some(imp_module) = self.modules.get(&imp_idx) {
                let imp_ast = AstModule {
                    name: imp_module.name,
                    path: imp_module.path.clone(),
                    stmts: imp_module.parse_result.statements.as_slice(),
                };
                self.dep_graph
                    .borrow_mut()
                    .extract_edges_for_module(imp_idx, &imp_ast, &self.pool);
                self.register_implicit_imports(imp_idx);
            }
        }

        let mut lowerer = self.make_lowerer(&parsed);
        lowerer.lower_module_prototypes(&parsed.parse_result.statements, module_idx);

        let lowered = lowerer.lower_module_bodies(&parsed.parse_result.statements, module_idx);
        for err in lowerer.lowering_errors().iter() {
            reporter.add_type_error(TypeError {
                kind: TypeErrorKind::Generic(err.0.clone()),
                span: err.1,
            });
        }

        self.modules.insert(module_idx, parsed);
        self.hir_modules.insert(module_idx, lowered);

        if reporter.has_errors() {
            return reporter;
        }

        self.type_checker
            .borrow_mut()
            .check_attributes(&lowerer.ctx.attrs.borrow());

        let order = self.dep_graph.borrow().get_module_compilation_order();
        let rank: FxHashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &m)| (m, i)).collect();
        let mut invalidation_set: Vec<usize> = self
            .dep_graph
            .borrow()
            .reverse_deps_transitive_modules(module_idx)
            .into_iter()
            .collect();
        if !invalidation_set.contains(&module_idx) {
            invalidation_set.push(module_idx);
        }
        invalidation_set.sort_by_key(|m| rank.get(m).copied().unwrap_or(usize::MAX));

        {
            let mut checker = self.type_checker.borrow_mut();

            for &m in &invalidation_set {
                checker.forget_module(m);
            }

            for &m in &invalidation_set {
                if let Some(hir) = self.hir_modules.get(&m) {
                    checker.register_module(hir, m);
                }
            }
        }

        let updated: Vec<(usize, &HirModule<'a, 'bump>)> = invalidation_set
            .iter()
            .filter_map(|&m| self.hir_modules.get(&m).map(|h| (m, h)))
            .collect();
        self.cpy_ctx.borrow_mut().recompute(&updated);

        {
            let mut checker = self.type_checker.borrow_mut();
            for &m in &invalidation_set {
                if let Some(hir) = self.hir_modules.get(&m) {
                    checker.check_module_body(hir, m);
                }
            }
            for err in checker.take_errors() {
                reporter.add_type_error(err);
            }
        }
        reporter
    }

    pub fn bootstrap_stdlib(
        &mut self,
        stdlib_path: &Path,
    ) -> Result<ErrorReporter<'a>, BuildError<'a>> {
        self.source_roots.push(stdlib_path.to_path_buf());

        let stdlib_files = crate::file_handling::collect_zeta_files(stdlib_path)?;
        let mut reporter = ErrorReporter::new();
        let mut batch: Vec<(usize, StrId, ModuleWithArena<'a, 'bump>)> = Vec::new();

        for path in &stdlib_files {
            let canonical_name = self.canonical_key(path);
            let module_idx = self.module_idx_for(canonical_name);

            let contents = std::fs::read_to_string(path)
                .map_err(|e| BuildError::FailedToReadFile(path.clone(), e))?;
            let parsed = crate::file_handling::parse_single_file_from_source(
                self.pool.clone(),
                PathBuf::from(canonical_name.as_str()),
                contents,
            )?;
            self.loaded_sources
                .insert(canonical_name.to_string(), parsed.source.to_string());

            if parsed.parse_result.diagnostics.has_errors() {
                for perr in &parsed.parse_result.diagnostics.errors {
                    reporter.add_parser_error(perr.clone());
                }
                continue;
            }

            self.stdlib_module_ids.insert(module_idx);

            let ast_module = codex_dependency_graph::dep_graph::AstModule {
                name: canonical_name,
                path: parsed.path.clone(),
                stmts: parsed.parse_result.statements.as_slice(),
            };
            self.dep_graph.borrow_mut().register_module_structure(
                module_idx,
                &ast_module,
                &self.pool,
            );
            batch.push((module_idx, canonical_name, parsed));
        }

        for (module_idx, name, parsed) in &batch {
            let ast_module = codex_dependency_graph::dep_graph::AstModule {
                name: *name,
                path: parsed.path.clone(),
                stmts: parsed.parse_result.statements.as_slice(),
            };
            self.dep_graph.borrow_mut().extract_edges_for_module(
                *module_idx,
                &ast_module,
                &self.pool,
            );
        }

        let order = self.dep_graph.borrow().get_module_compilation_order();
        let mut pending: FxHashMap<usize, (StrId, ModuleWithArena<'a, 'bump>)> = batch
            .into_iter()
            .map(|(idx, name, parsed)| (idx, (name, parsed)))
            .collect();

        self.compile_pending_grouped(&order, &mut pending, &mut reporter);

        for (module_idx, (_name, parsed)) in pending {
            reporter.merge(self.lower_and_check_module(module_idx, parsed));
        }

        let mismatches = self
            .dep_graph
            .borrow()
            .check_package_paths(&[stdlib_path.to_path_buf()]);
        for m in &mismatches {
            eprintln!("error: {}", m);
        }
        let unresolved = self.dep_graph.borrow().unresolved_import_messages();
        for msg in &unresolved {
            eprintln!("error: {}", msg);
        }
        if !mismatches.is_empty() || !unresolved.is_empty() {
            return Err(BuildError::InvalidModuleStructure {
                package_mismatches: mismatches.len(),
                unresolved_imports: unresolved.len(),
            });
        }

        Ok(reporter)
    }

    fn make_reporter(&self) -> ErrorReporter<'a> {
        let mut reporter = ErrorReporter::new();

        for (path, source) in &self.loaded_sources {
            reporter.add_source_file(path.clone(), source.clone());
        }

        reporter
    }

    fn canonical_key(&self, path: &Path) -> StrId {
        let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        StrId(self.pool.intern(&p.to_string_lossy()))
    }

    pub fn is_in_source_roots(&self, path: &Path) -> bool {
        let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.source_roots.iter().any(|r| {
            let r = std::fs::canonicalize(r).unwrap_or_else(|_| r.clone());
            p.starts_with(r)
        })
    }

    pub fn remove_module(&mut self, path: &Path) -> ErrorReporter<'a> {
        let name = self.canonical_key(path);
        let Some(&idx) = self.module_ids.get(&name) else {
            return self.make_reporter();
        };

        // must be computed BEFORE the graph forgets the module
        let order = self.dep_graph.borrow().get_module_compilation_order();
        let rank: FxHashMap<usize, usize> =
            order.iter().enumerate().map(|(i, &m)| (m, i)).collect();
        let mut dependents: Vec<usize> = self
            .dep_graph
            .borrow()
            .reverse_deps_transitive_modules(idx)
            .into_iter()
            .filter(|&m| m != idx)
            .collect();
        dependents.sort_by_key(|m| rank.get(m).copied().unwrap_or(usize::MAX));

        self.dep_graph.borrow_mut().remove_module(idx);
        self.type_checker.borrow_mut().forget_module(idx);
        self.cpy_ctx.borrow_mut().remove_module(idx);
        self.module_ids.remove(&name);
        self.stdlib_module_ids.remove(&idx);
        self.loaded_sources.remove(name.as_str());
        self.pending_mono.retain(|&m| m != idx);
        self.hir_modules.remove(&idx);
        self.codegen_hir_modules.remove(&idx);
        if let Some(old) = self.modules.remove(&idx) {
            self.retired_modules.push(old);
        }

        let mut reporter = self.make_reporter();

        for &d in &dependents {
            if let Some(m) = self.modules.get(&d) {
                let ast = AstModule {
                    name: m.name,
                    path: m.path.clone(),
                    stmts: m.parse_result.statements.as_slice(),
                };
                self.dep_graph
                    .borrow_mut()
                    .extract_edges_for_module(d, &ast, &self.pool);
            }
        }

        {
            let mut checker = self.type_checker.borrow_mut();
            for &m in &dependents {
                checker.forget_module(m);
            }
            for &m in &dependents {
                if let Some(h) = self.hir_modules.get(&m) {
                    checker.register_module(h, m);
                }
            }
        }
        let updated: Vec<(usize, &HirModule<'a, 'bump>)> = dependents
            .iter()
            .filter_map(|&m| self.hir_modules.get(&m).map(|h| (m, h)))
            .collect();
        self.cpy_ctx.borrow_mut().recompute(&updated);
        {
            let mut checker = self.type_checker.borrow_mut();
            for &m in &dependents {
                if let Some(h) = self.hir_modules.get(&m) {
                    checker.check_module_body(h, m);
                }
            }
            for err in checker.take_errors() {
                reporter.add_type_error(err);
            }
        }
        reporter
    }

    fn first_ctx_structs_lookup(&self, name: &StrId) -> Option<HirStruct<'a, 'bump>> {
        self.registry.structs.borrow().get(name).copied()
    }

    fn first_ctx_enums_lookup(&self, name: &StrId) -> Option<HirEnum<'a, 'bump>> {
        self.registry.enums.borrow().get(name).copied()
    }

    fn force_instantiate_allocator_frees(&self) {
        let scratch_bump = self.scratch_bump_ref();

        let mut scratch_lowerer = HirLowerer::new(
            self.pool.clone(),
            scratch_bump,
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );

        let mut monomorphizer = Monomorphizer::new(
            self.pool.clone(),
            &scratch_bump,
            self.registry.functions.clone(),
            &mut scratch_lowerer.ctx,
            self.registry.instantiated_functions.clone(),
            self.registry.instantiated_structs.clone(),
            self.registry.instantiated_struct_origins.clone(),
            self.registry.instantiated_enums.clone(),
            self.registry.instantiated_enum_origins.clone(),
            self.env_structs.clone(),
        );

        monomorphizer.force_instantiate_allocator_frees();
    }

    fn force_instantiate_drops(&self) {
        let scratch_bump = self.scratch_bump_ref();

        let mut scratch_lowerer = HirLowerer::new(
            self.pool.clone(),
            scratch_bump,
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );

        let mut monomorphizer = Monomorphizer::new(
            self.pool.clone(),
            &scratch_lowerer.ctx.bump,
            self.registry.functions.clone(),
            &mut scratch_lowerer.ctx,
            self.registry.instantiated_functions.clone(),
            self.registry.instantiated_structs.clone(),
            self.registry.instantiated_struct_origins.clone(),
            self.registry.instantiated_enums.clone(),
            self.registry.instantiated_enum_origins.clone(),
            self.env_structs.clone(),
        );

        monomorphizer.force_instantiate_drops();
    }

    /// SAFETY: the Box's heap allocation is stable across moves of `Compiler`, and
    /// `GrowableBump` only mutates through its own interior mutability, so
    /// handing out a `&'bump` ref derived from `&self` is sound.
    fn scratch_bump_ref(&self) -> &'bump GrowableBump<'bump> {
        unsafe { &*(self.lowerer_bump.as_ref() as *const GrowableBump<'bump>) }
    }
}

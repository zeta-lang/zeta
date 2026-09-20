#![feature(allocator_api)]

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;

use codex_dependency_graph::dep_graph::{AstModule, DepGraph};
use engraver_assembly_emit::backend::Backend;
use engraver_assembly_emit::cranelift::cranelift_backend::CraneliftBackend;
use ir::analysis_context::CopyAnalysisCtx;
use ir::ast::Stmt;
use ir::auto_imports::AutoImportRegistry;
use ir::errors::reporter::ErrorReporter;
use ir::errors::type_error::{TypeError, TypeErrorKind};
use ir::hir::{Hir, HirEnum, HirModule, HirStruct, StrId};
use ir::ir_hasher::{FxHashMap, HashMap, HashSet};
use ir::registry::global_registry::GlobalRegistry;
use scribe_parser::hir_lowerer::HirLowerer;
use scribe_parser::hir_lowerer::lambda_hoisting::LambdaHoister;
use scribe_parser::hir_lowerer::monomorphization::Monomorphizer;
use sentinel_typechecker::TypeChecker;
use zetaruntime::arena::GrowableAtomicBump;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub mod compilation_passes;
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

pub struct Compiler<'a, 'bump> {
    pool: Arc<StringPool>,
    registry: GlobalRegistry<'a, 'bump>,
    auto_imports: Rc<RefCell<AutoImportRegistry>>,

    dep_graph: &'a RefCell<DepGraph>,
    #[allow(unused)] // Avoids a UB
    dep_graph_storage: Box<RefCell<DepGraph>>,
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

        let cpy_ctx = Rc::new(RefCell::new(CopyAnalysisCtx::new(
            &[],
            registry.clone(),
            pool.clone(),
        )));
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
        })
    }

    pub fn dep_graph(&self) -> &'a RefCell<DepGraph> {
        self.dep_graph
    }

    pub fn path_for_module(&self, module_idx: usize) -> Option<&Path> {
        self.modules.get(&module_idx).map(|m| m.path.as_path())
    }

    pub fn module_idx_for_path(&self, path: &Path) -> Option<usize> {
        let name = StrId(self.pool.intern(path.to_string_lossy().as_ref()));
        self.module_ids.get(&name).copied()
    }

    pub fn type_checker(&self) -> Rc<RefCell<TypeChecker<'a, 'bump>>> {
        self.type_checker.clone()
    }

    pub fn dep_graph_ref(&self) -> &'a RefCell<DepGraph> {
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
        self.source_roots.push(dir.to_path_buf());

        let files = crate::file_handling::collect_zeta_files(dir)?;
        let sources = loader
            .load_files(&files)
            .map_err(|e| BuildError::FailedToReadFile(dir.to_path_buf(), e))?;

        let mut reporter = ErrorReporter::new();
        let mut batch: Vec<(usize, StrId, ModuleWithArena<'a, 'bump>)> = Vec::new();

        for file in sources {
            let canonical_name = file
                .path
                .to_str()
                .ok_or_else(|| BuildError::InvalidFileName(Vec::new()))?;
            let name = StrId(self.pool.intern(canonical_name));
            let module_idx = self.module_idx_for(name);

            let parsed = crate::file_handling::parse_single_file_from_source(
                self.pool.clone(),
                file.path,
                file.source,
            )?;

            self.loaded_sources
                .insert(name.to_string(), parsed.source.to_string());

            for perr in &parsed.parser_diagnostics.errors {
                reporter.add_parser_error(perr.clone());
            }

            if is_stdlib {
                self.stdlib_module_ids.insert(module_idx);
            }

            let ast_module = codex_dependency_graph::dep_graph::AstModule {
                name,
                path: parsed.path.clone(),
                stmts: parsed.stmts.as_slice(),
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
                stmts: parsed.stmts.as_slice(),
            };
            self.dep_graph.borrow_mut().extract_edges_for_module(
                *module_idx,
                &ast_module,
                &self.pool,
            );
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
            .check_package_paths(&[dir.to_path_buf()], &self.pool);
        for m in &mismatches {
            eprintln!("error: {}", m);
        }

        let unresolved = self
            .dep_graph
            .borrow()
            .unresolved_import_messages(&self.pool);
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
            parsed.bump.clone(),
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
        let hir = lowerer.lower_module_bodies(&parsed.stmts, module_idx);

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
                .map(|m| m.bump.clone())
                .expect("Expected module idx to exist");
            let new_module = HirModule {
                name: existing.name,
                imports: existing.imports,
                items: bump.alloc_slice(&merged),
            };
            self.codegen_hir_modules.insert(module_idx, new_module);
        }
    }

    fn monomorphize_module(&mut self, module_idx: usize, lowerer: &mut HirLowerer<'a, 'bump>) {
        let checked_hir = self.hir_modules[&module_idx];
        let bump = &self.modules.get(&module_idx).unwrap().bump;

        let hoister = LambdaHoister::new(bump.clone(), self.pool.clone(), checked_hir.name);
        let hoisted_module = hoister.run(checked_hir);

        let mut monomorphizer = Monomorphizer::new(
            self.pool.clone(),
            bump.clone(),
            self.registry.functions.clone(),
            &mut lowerer.ctx,
            self.registry.instantiated_functions.clone(),
            self.registry.instantiated_structs.clone(),
            self.registry.instantiated_struct_origins.clone(),
            self.registry.instantiated_enums.clone(),
            self.registry.instantiated_enum_origins.clone(),
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

        let shared_bump = pending.values().next().unwrap().1.bump.clone();
        let mut lowerer = HirLowerer::new(
            self.pool.clone(),
            shared_bump,
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );

        let module_stmts: FxHashMap<usize, Vec<Stmt<'a, 'bump>, Arc<GrowableAtomicBump<'bump>>>> =
            pending
                .iter()
                .map(|(&idx, (_name, parsed))| (idx, parsed.stmts.clone()))
                .collect();

        let mut lowered = lowerer.lower_all_modules(&module_stmts, order);

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
                self.monomorphize_module(module_idx, &mut lowerer);
            }
        }
    }

    fn lower_and_check_module(
        &mut self,
        module_idx: usize,
        parsed: ModuleWithArena<'a, 'bump>,
    ) -> ErrorReporter<'a> {
        let mut reporter = self.make_reporter();
        let mut lowerer = self.make_lowerer(&parsed);
        lowerer.lower_module_prototypes(&parsed.stmts, module_idx);

        let Some(mut lowerer) =
            self.lower_module_bodies_phase(module_idx, lowerer, parsed, &mut reporter)
        else {
            return reporter;
        };

        let invalidation_set = self
            .dep_graph
            .borrow()
            .reverse_deps_transitive_modules(module_idx);

        {
            let mut checker = self.type_checker.borrow_mut();
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

        if reporter.has_errors() {
            return reporter;
        }

        self.monomorphize_module(module_idx, &mut lowerer);
        reporter
    }

    pub fn emit(
        &mut self,
        out_dir: &Path,
        optimize: bool,
        verbose: bool,
        emit_obj: bool,
    ) -> Result<PathBuf, BuildError<'a>> {
        self.finalize_monomorphization();
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

        let mismatches = self
            .dep_graph
            .borrow()
            .check_package_paths(&self.source_roots, &self.pool);
        if !mismatches.is_empty() {
            return Err(BuildError::CompilationAborted {
                reason: mismatches
                    .iter()
                    .map(|m| m.to_string())
                    .collect::<Vec<_>>()
                    .join("\n"),
            });
        }

        let unresolved = self
            .dep_graph
            .borrow()
            .unresolved_import_messages(&self.pool);
        if !unresolved.is_empty() {
            return Err(BuildError::CompilationAborted {
                reason: unresolved.join("\n"),
            });
        }

        let compilation_order = self.dep_graph.borrow().get_module_compilation_order();

        let ordered_hir: Vec<HirModule<'a, 'bump>> = (0..self.next_module_idx)
            .filter_map(|i| self.codegen_hir_modules.get(&i).copied())
            .collect();

        let mut backend: CraneliftBackend =
            CraneliftBackend::new(self.pool.clone(), optimize, verbose);

        crate::file_handling::emit_all(
            &ordered_hir,
            &compilation_order,
            &mut backend,
            self.pool.clone(),
            Rc::new(crate::file_handling::collect_extern_c_names(&ordered_hir)),
            self.dep_graph,
            self.registry.clone(),
        );

        let out_obj = backend
            .finish(&out_dir.to_path_buf())
            .map_err(|e| BuildError::FinishError(Box::new(e)))?;

        if emit_obj {
            Ok(out_obj)
        } else {
            let program_path = out_dir.join("program");
            crate::link::link(
                &[out_obj.to_str().unwrap()],
                program_path.to_str().unwrap(),
                true,
            )?;
            Ok(program_path)
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
        let canonical_name = StrId(self.pool.intern(&path.to_string_lossy()));
        let module_idx = self.module_idx_for(canonical_name);

        let mut reporter = self.make_reporter();

        let parsed = match crate::file_handling::parse_single_file_from_source(
            self.pool.clone(),
            PathBuf::from(canonical_name.as_str()),
            source,
        ) {
            Ok(m) => m,
            Err(e) => {
                eprintln!("error: {}", e);
                return reporter;
            }
        };

        for perr in &parsed.parser_diagnostics.errors {
            reporter.add_parser_error(perr.clone());
        }
        reporter.add_source_file(canonical_name.to_string(), parsed.source.to_string());
        if parsed.parser_diagnostics.has_errors() {
            return reporter;
        }

        let mut lowerer = HirLowerer::new(
            self.pool.clone(),
            parsed.bump.clone(),
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );
        let hir = lowerer.lower_module(&parsed.stmts, module_idx);
        for err in lowerer.lowering_errors().iter() {
            reporter.add_type_error(TypeError {
                kind: TypeErrorKind::Generic(err.0.clone()),
                span: err.1,
            });
        }

        let ast_module = AstModule {
            name: canonical_name,
            path: parsed.path.clone(),
            stmts: parsed.stmts.as_slice(),
        };

        let importers = self.dep_graph.borrow().get_module_importers(module_idx);

        self.dep_graph
            .borrow_mut()
            .update_module_items(module_idx, &ast_module, &self.pool);

        for imp_idx in importers {
            if let Some(imp_module) = self.modules.get(&imp_idx) {
                let imp_ast = codex_dependency_graph::dep_graph::AstModule {
                    name: imp_module.name,
                    path: parsed.path.clone(),
                    stmts: imp_module.stmts.as_slice(),
                };
                self.dep_graph
                    .borrow_mut()
                    .extract_edges_for_module(imp_idx, &imp_ast, &self.pool);
            }
        }

        self.modules.insert(module_idx, parsed);
        self.hir_modules.insert(module_idx, hir);

        let invalidation_set = self
            .dep_graph
            .borrow()
            .reverse_deps_transitive_modules(module_idx);

        {
            let mut checker = self.type_checker.borrow_mut();
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
                reporter.add_type_error(err.clone());
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
            let canonical_name = path
                .to_str()
                .ok_or_else(|| BuildError::InvalidFileName(Vec::new()))?;
            let name = StrId(self.pool.intern(canonical_name));
            let module_idx = self.module_idx_for(name);

            let parsed = crate::file_handling::parse_single_file(self.pool.clone(), path.clone())?;

            self.loaded_sources.insert(
                parsed.path.to_string_lossy().to_string(),
                parsed.source.to_string(),
            );

            if parsed.parser_diagnostics.has_errors() {
                for perr in &parsed.parser_diagnostics.errors {
                    reporter.add_parser_error(perr.clone());
                }
                continue;
            }

            self.stdlib_module_ids.insert(module_idx);

            let ast_module = codex_dependency_graph::dep_graph::AstModule {
                name,
                path: parsed.path.clone(),
                stmts: parsed.stmts.as_slice(),
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
                stmts: parsed.stmts.as_slice(),
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
            .check_package_paths(&[stdlib_path.to_path_buf()], &self.pool);
        for m in &mismatches {
            eprintln!("error: {}", m);
        }
        let unresolved = self
            .dep_graph
            .borrow()
            .unresolved_import_messages(&self.pool);
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

    pub fn close_module(&mut self, name: &str) {
        let canonical_name = StrId(self.pool.intern(name));
        if let Some(&idx) = self.module_ids.get(&canonical_name) {
            self.modules.remove(&idx);
            self.hir_modules.remove(&idx);
            self.cpy_ctx.borrow_mut().remove_module(idx);
        }
    }

    fn first_ctx_structs_lookup(&self, name: &StrId) -> Option<HirStruct<'a, 'bump>> {
        self.registry.structs.borrow().get(name).copied()
    }

    fn first_ctx_enums_lookup(&self, name: &StrId) -> Option<HirEnum<'a, 'bump>> {
        self.registry.enums.borrow().get(name).copied()
    }

    fn force_instantiate_allocator_frees(&self) {
        let Some(scratch_bump) = self.modules.values().next().map(|m| m.bump.clone()) else {
            return;
        };

        let mut scratch_lowerer = HirLowerer::new(
            self.pool.clone(),
            scratch_bump.clone(),
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );

        let monomorphizer = Monomorphizer::new(
            self.pool.clone(),
            scratch_bump,
            self.registry.functions.clone(),
            &mut scratch_lowerer.ctx,
            self.registry.instantiated_functions.clone(),
            self.registry.instantiated_structs.clone(),
            self.registry.instantiated_struct_origins.clone(),
            self.registry.instantiated_enums.clone(),
            self.registry.instantiated_enum_origins.clone(),
        );

        monomorphizer.force_instantiate_allocator_frees();
    }

    fn force_instantiate_drops(&self) {
        let Some(scratch_bump) = self.modules.values().next().map(|m| m.bump.clone()) else {
            return;
        };

        let mut scratch_lowerer = HirLowerer::new(
            self.pool.clone(),
            scratch_bump.clone(),
            self.dep_graph,
            self.registry.clone(),
            self.auto_imports.clone(),
        );

        let monomorphizer = Monomorphizer::new(
            self.pool.clone(),
            scratch_bump,
            self.registry.functions.clone(),
            &mut scratch_lowerer.ctx,
            self.registry.instantiated_functions.clone(),
            self.registry.instantiated_structs.clone(),
            self.registry.instantiated_struct_origins.clone(),
            self.registry.instantiated_enums.clone(),
            self.registry.instantiated_enum_origins.clone(),
        );

        monomorphizer.force_instantiate_drops();
    }
}

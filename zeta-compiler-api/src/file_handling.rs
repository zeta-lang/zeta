use crate::main_structs::{CompilerError, ModuleWithArena};
use codex_dependency_graph::dep_graph::DepGraph;
use emberforge_compiler::midend::copy_analysis::drop_glue::DropGlueRegistry;
use emberforge_compiler::midend::ir::module_lowerer::MirModuleLowerer;
use engraver_assembly_emit::backend::Backend;
use engraver_assembly_emit::cranelift::cranelift_backend::CraneliftBackend;
use ir::hir::{HirModule, StrId};
use ir::ir_hasher::HashSet;
use ir::registry::global_registry::GlobalRegistry;
use ir::ssa_ir::Module;
use scribe_parser::parser::ParseResult;
use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;
use walkdir::WalkDir;
use zetaruntime::bump::GrowableBump;
use zetaruntime::string_pool::StringPool;

pub fn compiler_lib_path<'a>() -> Result<PathBuf, CompilerError<'a>> {
    let exe = std::env::current_exe()
        .map_err(|_| CompilerError::SourceNotFound(PathBuf::from("current_exe()")))?;

    let root = exe
        .parent()
        .ok_or_else(|| CompilerError::SourceNotFound(PathBuf::from("compiler root")))?
        .to_path_buf();

    Ok(root.join("lib"))
}

pub(crate) fn collect_zeta_files<'a>(dir: &Path) -> Result<Vec<PathBuf>, CompilerError<'a>> {
    if !dir.exists() {
        return Err(CompilerError::SourceNotFound(dir.to_path_buf()));
    }

    let files = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|e| e.to_str()) == Some("zeta"))
        .map(|e| e.into_path())
        .collect::<Vec<_>>();

    if files.is_empty() {
        eprintln!(
            "hit `files.is_empty()`, iter: {:?}",
            WalkDir::new(dir).into_iter().collect::<Vec<_>>()
        );
        return Err(CompilerError::NoSourceFiles);
    }

    Ok(files)
}

pub fn collect_extern_c_names<'a, 'bump>(
    hir_modules: &[ir::hir::HirModule<'a, 'bump>],
) -> HashSet<StrId> {
    let mut externs = HashSet::default();
    for module in hir_modules {
        for item in module.items {
            if let ir::hir::Hir::Func(f) = item
                && matches!(
                    f.function_metadata.extern_modifier,
                    ir::ast::ExternModifier::Abi(_)
                )
                && f.body.is_none()
            {
                externs.insert(f.name);
            }
        }
    }
    externs
}

pub fn parse_single_file_from_source<'a, 'bump>(
    pool: Arc<StringPool>,
    name: PathBuf,
    source: String,
    bump: &'bump GrowableBump,
) -> Result<ModuleWithArena<'a, 'bump>, CompilerError<'a>>
where
    'a: 'bump,
    'bump: 'a,
{
    parse_single_module(pool, name, source, bump)
}

fn parse_single_module<'a, 'bump>(
    pool: Arc<StringPool>,
    path: PathBuf,
    contents: String,
    bump: &'bump GrowableBump,
) -> Result<ModuleWithArena<'a, 'bump>, CompilerError<'a>>
where
    'a: 'bump,
    'bump: 'a,
{
    let file_name_str = path
        .to_str()
        .ok_or_else(|| CompilerError::InvalidFileName(Vec::new()))?;

    let name = StrId(pool.intern(file_name_str));

    if contents.is_empty() {
        return Ok(ModuleWithArena {
            name,
            path,
            parse_result: ParseResult {
                statements: Vec::new(),
                diagnostics: scribe_parser::parser::ParserDiagnostics::new(),
            },
            source: String::new(),
            bump,
        });
    }

    let parse_result: ParseResult<'a, 'bump> =
        scribe_parser::parser::parse_program(contents.as_str(), name.as_str(), pool.clone(), bump);

    Ok(ModuleWithArena {
        name,
        path,
        parse_result,
        source: contents,
        bump,
    })
}

pub(crate) fn emit_all<'a, 'bump>(
    hir_modules: &[HirModule<'a, 'bump>],
    compilation_order: &[usize],
    pool: Arc<StringPool>,
    extern_c_names: Rc<HashSet<StrId>>,
    dep_graph: &'a RefCell<DepGraph<'a>>,
    registry: GlobalRegistry<'a, 'bump>,
    optimize: bool,
    verbose: bool,
    bump: &'bump GrowableBump,
) -> (Instant, CraneliftBackend)
where
    'bump: 'a,
{
    let codegen_start = Instant::now();

    let glue_registry = DropGlueRegistry::new(registry.clone(), pool.clone());

    let mir_module: Module = MirModuleLowerer::new(
        pool.clone(),
        extern_c_names.clone(),
        dep_graph,
        glue_registry,
        registry.instantiated_functions.clone(),
        registry.struct_methods.clone(),
        registry.clone(),
        bump,
    )
    .lower_all_modules(hir_modules, compilation_order);

    let mut backend: CraneliftBackend = CraneliftBackend::new(pool.clone(), optimize, verbose);

    CraneliftBackend::emit_module(&mut backend, &mir_module);
    (codegen_start, backend)
}

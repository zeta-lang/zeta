use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

use clap::{Parser, Subcommand};
use codex_dependency_graph::dep_graph::DepGraph;
use mimalloc::MiMalloc;
use zeta_compiler_api::Compiler;

use zeta_compiler_api::file_handling::compiler_lib_path;
use zeta_compiler_api::file_loader::choose_file_loader;
use zetaruntime::bump::GrowableBump;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[arg(short, long, global = true)]
    verbose: bool,

    #[arg(long, global = true, default_value = "true")]
    link_libc: bool,

    #[arg(long, global = true)]
    lib: Option<PathBuf>,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build Zeta source files
    Build {
        path: Option<PathBuf>,

        #[arg(long, default_value_os = "./build")]
        out_dir: PathBuf,

        #[arg(short, long)]
        optimize: bool,

        #[arg(long)]
        emit_obj: bool,
    },

    /// Run Zeta source files directly
    Run {
        path: Option<PathBuf>,

        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Type check and borrow check without generating code
    Check { path: Option<PathBuf> },
}

fn main() -> ExitCode {
    let cli = Cli::parse();

    match run(cli) {
        Ok(code) => code,
        Err(message) => {
            eprintln!("error: {message}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<ExitCode, String> {
    let verbose = cli.verbose;
    let lib_override = cli.lib;

    match cli.command {
        Commands::Build {
            path,
            out_dir,
            optimize,
            emit_obj,
        } => {
            if verbose {
                let source_path = path.clone().unwrap_or_else(|| PathBuf::from("./src"));
                println!("Building from: {}", source_path.display());
                println!("Output directory: {}", out_dir.display());
                println!("Optimization: {}", optimize);
            }
            run_compiler(path, &out_dir, optimize, verbose, emit_obj, lib_override)?;
            Ok(ExitCode::SUCCESS)
        }

        Commands::Run { path, args } => {
            let out_dir = PathBuf::from("./build");
            if verbose {
                let source_path = path.clone().unwrap_or_else(|| PathBuf::from("./src"));
                println!("Running from: {}", source_path.display());
                if !args.is_empty() {
                    println!("Arguments: {:?}", args);
                }
            }

            let program_path = run_compiler(path, &out_dir, false, verbose, false, lib_override)?;

            if verbose {
                println!("Executing: {}", program_path.display());
            }

            let status = Command::new(&program_path)
                .args(&args)
                .status()
                .map_err(|e| format!("failed to execute {}: {e}", program_path.display()))?;

            Ok(if status.success() {
                ExitCode::SUCCESS
            } else {
                // Propagate the program's exit code; a signal kill has no code, so use 1.
                ExitCode::from(
                    status
                        .code()
                        .and_then(|c| u8::try_from(c).ok())
                        .unwrap_or(1),
                )
            })
        }

        Commands::Check { path } => {
            if verbose {
                let source_path = path.clone().unwrap_or_else(|| PathBuf::from("./src"));
                println!("Checking: {}", source_path.display());
            }
            check_compiler(path, lib_override)?;
            Ok(ExitCode::SUCCESS)
        }
    }
}

fn resolve_stdlib_path(lib_override: Option<PathBuf>) -> Result<PathBuf, String> {
    if let Some(custom) = lib_override {
        Ok(custom)
    } else {
        compiler_lib_path().map_err(|e| e.to_string())
    }
}

fn load_and_check<'a, 'bump>(
    compiler: &mut Compiler<'a, 'bump>,
    path: Option<PathBuf>,
    lib_override: Option<PathBuf>,
) -> Result<(), String>
where
    'bump: 'a,
    'a: 'bump,
{
    let stdlib_path = resolve_stdlib_path(lib_override)?;
    let source_path = path.unwrap_or_else(|| PathBuf::from("./src"));
    let file_loader = choose_file_loader();

    let stdlib_diags = compiler
        .load_directory(&file_loader, &stdlib_path, true)
        .map_err(|e| format!("could not load the standard library: {e}"))?;
    if stdlib_diags.has_errors() {
        stdlib_diags.report_all();
        return Err("the standard library has errors".into());
    }

    let user_diags = compiler
        .load_directory(&file_loader, &source_path, false)
        .map_err(|e| e.to_string())?;
    if user_diags.has_errors() {
        user_diags.report_all();
        return Err("type checking failed".into());
    }

    Ok(())
}

// No lifetime parameters: 'a and 'bump are now the local borrows, and no error leaves with them.
fn run_compiler(
    path: Option<PathBuf>,
    out_dir: &Path,
    optimize: bool,
    verbose: bool,
    emit_obj: bool,
    lib_override: Option<PathBuf>,
) -> Result<PathBuf, String> {
    // Arena first: locals drop in reverse order, so it outlives everything below.
    let bump = GrowableBump::new(4096, 8);
    let dep_graph = RefCell::new(DepGraph::new());

    let mut compiler = Compiler::new(&bump, &dep_graph)
        .map_err(|e| format!("failed to start the compiler: {e}"))?;

    load_and_check(&mut compiler, path, lib_override)?;
    compiler
        .emit(out_dir, optimize, verbose, emit_obj)
        .map_err(|e| e.to_string())
}

fn check_compiler(path: Option<PathBuf>, lib_override: Option<PathBuf>) -> Result<(), String> {
    let bump = GrowableBump::new(4096, 8);
    let dep_graph = RefCell::new(DepGraph::new());

    let mut compiler = Compiler::new(&bump, &dep_graph)
        .map_err(|e| format!("failed to start the compiler: {e}"))?;

    load_and_check(&mut compiler, path, lib_override)?;

    compiler.report_check_timings();
    println!("No errors found.");
    Ok(())
}

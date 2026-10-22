use std::path::PathBuf;
use std::process::Command;

use clap::{ArgMatches, CommandFactory, Error, FromArgMatches, Parser, Subcommand};
use mimalloc::MiMalloc;
use zeta_compiler_api::Compiler;

use zeta_compiler_api::file_handling::compiler_lib_path;
use zeta_compiler_api::file_loader::choose_file_loader;
use zeta_compiler_api::main_structs::CompilerError;

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

fn main() -> Result<(), CompilerError<'static>> {
    let mut matches: ArgMatches = Cli::command().get_matches();
    let cli_result: Result<Cli, Error> =
        Cli::from_arg_matches_mut(&mut matches).map_err(|err: clap::error::Error| {
            let mut cmd = Cli::command();
            err.format(&mut cmd)
        });

    let cli: Cli = match cli_result {
        Ok(cli) => cli,
        Err(error) => {
            error.print().expect("failed to print error message");
            std::process::exit(error.exit_code());
        }
    };

    let verbose = cli.verbose;

    let lib_override = cli.lib.clone();

    let result = match cli.command {
        Commands::Build {
            path,
            out_dir,
            optimize,
            emit_obj,
        } => {
            let source_path = path.clone().unwrap_or_else(|| PathBuf::from("./src"));
            if verbose {
                println!("Building from: {}", source_path.display());
                println!("Output directory: {}", out_dir.display());
                println!("Optimization: {}", optimize);
            }

            run_compiler(path, &out_dir, optimize, verbose, emit_obj, lib_override)?;
            Ok(())
        }
        Commands::Run { path, args } => {
            let source_path = path.clone().unwrap_or_else(|| PathBuf::from("./src"));
            let out_dir = PathBuf::from("./build");

            if verbose {
                println!("Running from: {}", source_path.display());

                if !args.is_empty() {
                    println!("Arguments: {:?}", args);
                }
            }

            let program_path = run_compiler(
                path,
                &out_dir,
                false,
                verbose,
                false, // always link for `run`
                lib_override,
            )?;

            if verbose {
                println!("Executing: {}", program_path.display());
            }

            let status = Command::new(&program_path)
                .args(&args)
                .status()
                .expect("failed to execute compiled Zeta program");

            if !status.success() {
                std::process::exit(status.code().unwrap_or(1));
            }

            return Ok(());
        }
        Commands::Check { path } => {
            let source_path = path.clone().unwrap_or_else(|| PathBuf::from("./src"));
            if verbose {
                println!("Checking: {}", source_path.display());
            }

            check_compiler(path, verbose, lib_override)?;
            Ok(())
        }
    };

    match result {
        Ok(_) => Ok(()),
        Err(e) => {
            eprintln!("Error: {}", e);
            Err(e)
        }
    }
}

fn installed_stdlib_path() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var_os("HOME")?);

    let stdlib = home.join(".local").join("share").join("zeta").join("lib");

    stdlib.is_dir().then_some(stdlib)
}

fn resolve_stdlib_path(lib_override: Option<PathBuf>) -> Result<PathBuf, CompilerError<'static>> {
    if let Some(custom) = lib_override {
        Ok(custom)
    } else if let Some(installed) = installed_stdlib_path() {
        Ok(installed)
    } else {
        compiler_lib_path()
    }
}

/// Loads the stdlib and the user's sources, running parsing, lowering and
/// type/borrow checking. Prints diagnostics and returns Err on failure.
fn load_and_check<'a, 'bump>(
    compiler: &mut Compiler<'a, 'bump>,
    path: Option<PathBuf>,
    lib_override: Option<PathBuf>,
) -> Result<(), CompilerError<'a>>
where
    'bump: 'a,
    'a: 'bump,
{
    let stdlib_path = resolve_stdlib_path(lib_override)?;
    let source_path = path.unwrap_or_else(|| PathBuf::from("./src"));
    let file_loader = choose_file_loader();

    let stdlib_diags = compiler
        .load_directory(&file_loader, &stdlib_path, true)
        .unwrap_or_else(|e| panic!("Could not load stdlib because of {e}"));

    if stdlib_diags.has_errors() {
        stdlib_diags.report_all();
        return Err(CompilerError::ParserError(vec![]));
    }

    let user_diags = compiler.load_directory(&file_loader, &source_path, false)?;

    if user_diags.has_errors() {
        user_diags.report_all();
        return Err(CompilerError::TypeCheckError);
    }

    Ok(())
}

fn run_compiler<'a, 'bump>(
    path: Option<PathBuf>,
    out_dir: &PathBuf,
    optimize: bool,
    verbose: bool,
    emit_obj: bool,
    lib_override: Option<PathBuf>,
) -> Result<PathBuf, CompilerError<'a>>
where
    'bump: 'a,
    'a: 'bump,
{
    let mut compiler = Compiler::new()?;
    load_and_check(&mut compiler, path, lib_override)?;
    compiler.emit(out_dir, optimize, verbose, emit_obj)
}

fn check_compiler<'a, 'bump>(
    path: Option<PathBuf>,
    verbose: bool,
    lib_override: Option<PathBuf>,
) -> Result<(), CompilerError<'a>>
where
    'bump: 'a,
    'a: 'bump,
{
    let mut compiler = Compiler::new()?;
    load_and_check(&mut compiler, path, lib_override)?;

    if verbose {
        compiler.report_check_timings();
    }
    println!("No errors found.");
    Ok(())
}

pub mod dispatch;
pub mod doc_comments;
pub mod docs;
pub mod lsp_diagnostics;
pub mod semantic_tokens;
pub mod server;
pub mod state;
pub mod text_utils;
pub mod urls;

pub mod implementations;
#[cfg(test)]
mod tests;

use std::{cell::RefCell, error::Error};

use clap::{Parser, Subcommand};
use codex_dependency_graph::dep_graph::DepGraph;
pub use server::Server;
use zetaruntime::bump::GrowableBump;

#[derive(Parser)]
#[command(
    name = "zeta-lsp",
    author,
    version,
    about = "Zeta Language Server",
    long_about = None
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand)]
pub enum Commands {
    /// Start the language server
    Lsp,

    /// Index documentation for Zed
    IndexDocs,

    /// Format source files
    Format {
        /// File or directory
        path: std::path::PathBuf,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command.unwrap_or(Commands::Lsp) {
        Commands::Lsp => {
            let dep_graph = RefCell::new(DepGraph::new());

            let lowerer_bump = GrowableBump::new(4096, 8);

            let compiler = zeta_compiler_api::Compiler::new(&lowerer_bump, &dep_graph)
                .map_err(|e| Box::new(e))
                .unwrap_or_else(|e| {
                    panic!("error while starting up compiler process for LSP: {e}")
                });
            let server = Server::new(compiler)?;
            server.run()?;
        }

        Commands::IndexDocs => {
            docs::index()?;
        }

        Commands::Format { path } => {
            println!("Formatting {:?}", path);
            // TODO
        }
    }

    Ok(())
}

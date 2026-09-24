use ir::hir::StrId;
use scribe_parser::parser::ParseResult;
use std::error::Error;
use std::fmt;
use std::io;
use std::path::PathBuf;
use zetaruntime::bump::GrowableBump;

#[derive(Clone, Debug)]
pub struct ModuleWithArena<'a, 'bump> {
    pub bump: Box<GrowableBump<'bump>>,
    pub name: StrId,
    pub path: PathBuf,
    pub parse_result: ParseResult<'a, 'bump>,
    pub source: String,
}

impl<'a, 'bump> ModuleWithArena<'a, 'bump> {
    /// SAFETY: `bump` is heap-allocated via `Box`; moving `ModuleWithArena`
    /// afterwards only moves the pointer, not the `GrowableBump` itself, so
    /// this reference stays valid for as long as the `Box` is kept alive.
    pub fn bump_ref(&self) -> &'bump GrowableBump<'bump> {
        unsafe { &*(self.bump.as_ref() as *const GrowableBump<'bump>) }
    }
}

#[derive(Debug)]
pub enum CompilerError<'a> {
    SourceNotFound(PathBuf),
    NoSourceFiles,
    FailedToReadFile(PathBuf, io::Error),
    FailedToAllocateBump,
    FailedToAllocateStringPool,
    InvalidFileName(Vec<u8>),
    ParserError(Vec<&'a str>),
    TypeCheckError,
    FinishError(Box<dyn std::error::Error>),
    LinkFailed,
    InvalidModuleStructure {
        package_mismatches: usize,
        unresolved_imports: usize,
    },
    CompilationAborted {
        reason: String,
    },
}

impl<'a> fmt::Display for CompilerError<'a> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompilerError::SourceNotFound(p) => write!(f, "Source not found: {}", p.display()),
            CompilerError::NoSourceFiles => write!(f, "No .zeta source files found"),
            CompilerError::FailedToReadFile(p, e) => {
                write!(f, "Failed to read {}: {}", p.display(), e)
            }
            CompilerError::FailedToAllocateBump => write!(f, "Failed to allocate bump arena"),
            CompilerError::FailedToAllocateStringPool => {
                write!(f, "Failed to allocate string pool")
            }
            CompilerError::InvalidFileName(b) => {
                write!(f, "Invalid file name bytes: {:?}", b)
            }
            CompilerError::ParserError(errs) => {
                write!(f, "Parser errors: {:?}", errs)
            }
            CompilerError::TypeCheckError => write!(f, "Type check failed"),
            CompilerError::FinishError(e) => write!(f, "Backend finish error: {}", e),
            CompilerError::LinkFailed => {
                write!(f, "Backend could not link the stdlib to the binary.")
            }
            CompilerError::InvalidModuleStructure {
                package_mismatches,
                unresolved_imports,
            } => write!(
                f,
                "invalid module structure {package_mismatches} {unresolved_imports}"
            ),
            CompilerError::CompilationAborted { reason } => {
                write!(f, "Compilation aborted: {reason}")
            }
        }
    }
}

impl<'a> Error for CompilerError<'a> {}

use std::path::PathBuf;
use std::process::{Command, ExitStatus};

use crate::main_structs::CompilerError;

pub fn link<'a>(objects: &[&str], output: &str, link_libc: bool) -> Result<(), CompilerError<'a>> {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    let driver = "gcc";

    #[cfg(target_os = "windows")]
    let driver = "gcc"; // MinGW; otherwise use cl.exe or clang

    let mut cmd = Command::new(driver);

    cmd.args(objects);

    // Look for the runtime in the user's installed Zeta directory.
    if let Some(home) = std::env::var_os("HOME") {
        let runtime = PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("zeta")
            .join("zeta_rt.c");

        if runtime.is_file() {
            cmd.arg(runtime);
        } else if std::path::Path::new("zeta_rt.c").is_file() {
            // Development fallback.
            cmd.arg("zeta_rt.c");
        }
    } else if std::path::Path::new("zeta_rt.c").is_file() {
        cmd.arg("zeta_rt.c");
    }

    cmd.arg("-o").arg(output);

    if !link_libc {
        cmd.arg("-nostdlib");
    }

    let status: ExitStatus = cmd.status().expect("failed to execute linker");

    if !status.success() {
        return Err(CompilerError::LinkFailed);
    }

    Ok(())
}

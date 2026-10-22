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
    if std::path::Path::new("zeta_rt.o").is_file() {
        cmd.arg("zeta_rt.o");
    } else if let Some(home) = std::env::var_os("HOME") {
        let zeta_share = PathBuf::from(home)
            .join(".local")
            .join("share")
            .join("zeta");
        
        let compiled_rt = zeta_share.as_path().join("zeta_rt.o");
        if compiled_rt.is_file() {
            cmd.arg(compiled_rt);
        }

        let uncompiled_rt = zeta_share.as_path().join("zeta_rt.c");
        if uncompiled_rt.is_file() {
            cmd.arg(uncompiled_rt);
        }
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

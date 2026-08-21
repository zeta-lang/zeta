use std::path::PathBuf;

#[cfg(target_os = "linux")]
use crate::io_uring_file_loader::IoUringFileLoader;

#[cfg(target_os = "windows")]
use crate::iocp_file_loader::IocpFileLoader;

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
use crate::std_file_loader::StdFileLoader;

pub trait FileLoader {
    fn load_files(&self, paths: &[PathBuf]) -> Result<Vec<SourceFile>, std::io::Error>;
}

#[derive(Clone)]
pub struct SourceFile {
    pub path: PathBuf,
    pub source: String,
}

pub fn choose_file_loader() -> impl FileLoader {
    #[cfg(target_os = "linux")]
    {
        return IoUringFileLoader::new(256);
    }

    #[cfg(not(any(target_os = "linux", target_os = "windows")))]
    {
        return StdFileLoader;
    }
}

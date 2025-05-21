use curl::easy::{Handler, WriteError};
use curl::{self};
use std::io::Cursor;
use std::path::{Path, PathBuf};

const ARTIFACTS: &[&str] = if cfg!(target_os = "macos") {
    &[
        "libConcreteRust.dylib",
        "libConcretelangRuntime.dylib",
        "libomp.dylib",
    ]
} else if cfg!(target_os = "linux") {
    &[
        "libConcreteRust.so",
        "libConcretelangRuntime.so",
        "libomp.so",
        "libhpx.so",
        "libhpx_core.so",
        "libhpx_iostreams.so",
    ]
} else {
    panic!("Unsupported platform");
};

const ARCHIVE: &str = if cfg!(all(target_os = "macos", target_arch = "x86_64")) {
    "cpu-binaries-macosx_x86_64.zip"
} else if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
    "cpu-binaries-macosx_arm64.zip"
} else if cfg!(all(target_os = "linux", target_arch = "x86_64")) {
    "cpu-binaries-linux-x86_64.zip"
} else {
    panic!("Unsupported platform");
};

const INSTALL_LOCK: &str = "install";
const INSTALLED_FILE: &str = "installed";
const URL: &str = "https://github.com/zama-ai/concrete/releases/download/v2.10.1-rc1";

include!("src/utils/flock.rs");
fn do_with_lock<F: FnMut()>(file: &Path, f: F) {
    let mut lock_file_name = file.to_path_buf();
    lock_file_name.set_extension("lock");
    let Ok(lock_file) = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(&lock_file_name)
    else {
        panic!("Failed to open lock file {}", lock_file_name.display())
    };
    let lock = FileLock::acquire(&lock_file).unwrap();
    let mut f = f;
    f();
    drop(lock);
}

fn install_artifacts(out_dir: &Path) {
    if !out_dir.join(INSTALLED_FILE).exists() {
        match std::env::var("COMPILER_BUILD_DIRECTORY") {
            Ok(v) => copy_local_artifacts(out_dir, PathBuf::from(v)),
            Err(_) => fetch_artifacts(out_dir),
        }
    }
}

fn copy_local_artifacts(out_dir: &Path, mut lib_dir: PathBuf) {
    println!(
        "cargo:warning=\"COMPILER_BUILD_DIRECTORY detected: building from local artifacts in {}\"",
        lib_dir.display()
    );
    lib_dir.push("lib");
    lib_dir = lib_dir
        .canonicalize()
        .map_err(|e| format!("Failed to canonicalize {}: {e}", lib_dir.display()))
        .unwrap();
    do_with_lock(&out_dir.join(INSTALL_LOCK), || {
        for art in ARTIFACTS {
            println!("cargo::rerun-if-changed={}", &lib_dir.join(art).display());
            std::fs::copy(&lib_dir.join(art), &out_dir.join(art))
                .map_err(|e| {
                    format!(
                        "Failed to copy {} to {}: {e}",
                        lib_dir.join(art).display(),
                        out_dir.join(art).display()
                    )
                })
                .unwrap();
        }
    });
}

fn fetch_artifacts(out_dir: &Path) {
    struct Archive(Vec<u8>);
    impl Handler for Archive {
        fn write(&mut self, data: &[u8]) -> Result<usize, WriteError> {
            self.0.extend_from_slice(data);
            Ok(data.len())
        }
    }

    do_with_lock(&out_dir.join(INSTALL_LOCK), || {
        if out_dir.join(INSTALLED_FILE).exists() {
            return;
        }
        let url = format!("{URL}/{ARCHIVE}");
        let mut handle = curl::easy::Easy2::new(Archive(Vec::new()));
        handle.get(true).unwrap();
        handle.follow_location(true).unwrap();
        handle.url(&url).unwrap();
        handle.perform().unwrap();
        assert_eq!(
            handle.response_code().unwrap(),
            200,
            "Failed to fetch the remote artifacts from {url}"
        );
        let mut zip = zip::ZipArchive::new(Cursor::new(handle.get_ref().0.as_slice())).unwrap();
        for i in 0..zip.len() {
            let mut content = zip.by_index(i).unwrap();
            let path = PathBuf::from(content.name());
            let mut file = std::fs::OpenOptions::new()
                .write(true)
                .create(true)
                .open(out_dir.join(path.file_name().unwrap()))
                .map_err(|e| {
                    format!(
                        "Failed to open file {}: {e}",
                        out_dir.join(path.file_name().unwrap()).display()
                    )
                })
                .unwrap();
            std::io::copy(&mut content, &mut file).unwrap();
        }
        std::fs::File::create(out_dir.join(INSTALLED_FILE))
            .map_err(|e| format!("Failed to create INSTALLED_FILE: {e}"))
            .unwrap();
    });
}

fn symlink_outdir(concrete_dir: &Path, out_dir: &Path) {
    if !concrete_dir.exists() {
        let _ = std::fs::create_dir(&concrete_dir);
    }
    if !out_dir.is_symlink() {
        assert!(
            std::fs::remove_dir(&out_dir).is_ok(),
            "Failed to delete original out_dir {}",
            out_dir.display()
        );
        std::os::unix::fs::symlink(concrete_dir, &out_dir)
            .map_err(|e| {
                format!(
                    "Failed to symlink {} to {}: {e}",
                    concrete_dir.display(),
                    out_dir.display()
                )
            })
            .unwrap();
    }
}

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir
        .join("../../../..")
        .canonicalize()
        .map_err(|e| format!("Failed to get target dir: {e}"))
        .unwrap();
    let concrete_dir = target_dir.join("concrete");

    symlink_outdir(&concrete_dir, &out_dir);
    install_artifacts(&out_dir);

    println!("cargo::rustc-link-search={}", out_dir.display());
    println!("cargo::metadata=build_dir={}", out_dir.display());
    println!("cargo::rustc-link-lib=dylib=ConcreteRust");
    println!("cargo::rustc-link-lib=z");
    #[cfg(target_os = "macos")]
    {
        println!("cargo::rustc-link-search=/opt/homebrew/lib");
        println!("cargo::rustc-link-lib=curses");
        println!("cargo::rustc-link-lib=zstd");
    }
    println!("cargo::rerun-if-changed=src");
}

#![feature(file_lock)]
use std::path::{Path, PathBuf};

#[cfg(target_os = "macos")]
const ARTIFACT_NAME: &str = "libConcreteRust.dylib";
#[cfg(target_os = "macos")]
const RUNTIME_NAME: &str = "libConcretelangRuntime.dylib";
#[cfg(target_os = "linux")]
const ARTIFACT_NAME: &str = "libConcreteRust.so";
#[cfg(target_os = "linux")]
const RUNTIME_NAME: &str = "libConcretelangRuntime.so";

fn do_with_lock<F: FnMut()>(file: &Path, f: F) {
    let mut lock_file_name = file.to_path_buf();
    lock_file_name.set_extension(format!(
        "{}.lock",
        lock_file_name
            .extension()
            .map_or("", |a| a.to_str().unwrap())
    ));
    let Ok(lock_file) = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .open(&lock_file_name)
    else {
        panic!("Failed to open lock file {}", lock_file_name.display())
    };
    assert!(
        lock_file.lock().is_ok(),
        "Failed to acquire lock on file {}",
        lock_file_name.display()
    );
    let mut f = f;
    f();
    assert!(
        lock_file.unlock().is_ok(),
        "Failed to release lock on file {}",
        lock_file_name.display()
    );
}

fn main() {
    let mut lib_dir = match std::env::var("COMPILER_BUILD_DIRECTORY") {
        Ok(v) => PathBuf::from(v),
        Err(_) => PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap())
            .join("../../../../build")
            .canonicalize()
            .unwrap(),
    };
    lib_dir.push("lib");
    lib_dir = lib_dir.canonicalize().unwrap();

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../../..").canonicalize().unwrap();
    let concrete_dir = target_dir.join("concrete");

    if !concrete_dir.exists() {
        let _ = std::fs::create_dir(&concrete_dir);
    }

    assert!(std::fs::remove_dir(&out_dir).is_ok(), "Failed to delete original out_dir {}", out_dir.display());
    std::os::unix::fs::symlink(concrete_dir, &out_dir).unwrap();

    do_with_lock(&out_dir.join(ARTIFACT_NAME), || {
        std::fs::copy(&lib_dir.join(ARTIFACT_NAME), &out_dir.join(ARTIFACT_NAME)).unwrap();
    });
    do_with_lock(&out_dir.join(RUNTIME_NAME), || {
        std::fs::copy(&lib_dir.join(RUNTIME_NAME), &out_dir.join(RUNTIME_NAME)).unwrap();
    });

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
    println!(
        "cargo::rerun-if-changed={}",
        lib_dir.join(ARTIFACT_NAME).display()
    );
}

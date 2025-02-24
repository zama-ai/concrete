use std::path::PathBuf;

#[cfg(target_os = "macos")]
const ARTIFACT_NAME: &str = "libConcreteRust.dylib";
#[cfg(target_os = "macos")]
const RUNTIME_NAME: &str = "libConcretelangRuntime.dylib";
#[cfg(target_os = "linux")]
const ARTIFACT_NAME: &str = "libConcreteRust.so";
#[cfg(target_os = "linux")]
const RUNTIME_NAME: &str = "libConcretelangRuntime.so";

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

    let target_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    std::fs::copy(&lib_dir.join(ARTIFACT_NAME), &target_dir.join(ARTIFACT_NAME)).unwrap();
    std::fs::copy(&lib_dir.join(RUNTIME_NAME), &target_dir.join(RUNTIME_NAME)).unwrap();
    println!(
        "cargo::rustc-link-search={}",
        target_dir.display()
    );
    println!("cargo::metadata=build_dir={}", target_dir.display());
    println!("cargo::rustc-link-lib=dylib=ConcreteRust");
    println!("cargo::rustc-link-lib=z");
    #[cfg(target_os = "macos")]
    {
        println!("cargo::rustc-link-search=/opt/homebrew/lib");
        println!("cargo::rustc-link-lib=curses");
        println!("cargo::rustc-link-lib=zstd");
    }
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed={}", lib_dir.join(ARTIFACT_NAME).display());
}

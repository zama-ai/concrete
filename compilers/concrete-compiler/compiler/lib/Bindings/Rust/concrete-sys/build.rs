use std::path::PathBuf;

#[cfg(target_os = "macos")]
const ARTIFACT_NAME: &str = "libConcreteSys.dylib";
#[cfg(target_os = "linux")]
const ARTIFACT_NAME: &str = "libConcreteSys.so";

fn main() {
    let mut from = PathBuf::from(
        std::env::var("COMPILER_BUILD_DIRECTORY")
            .expect("Environment variable COMPILER_BUILD_DIRECTORY must be set"),
    );
    from.push("lib");
    from.push(ARTIFACT_NAME);
    let from = from.canonicalize().unwrap();
    let mut to = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    to.push(ARTIFACT_NAME);
    std::fs::copy(&from, &to).unwrap();
    println!(
        "cargo::rustc-link-search={}",
        std::env::var("OUT_DIR").unwrap()
    );
    println!("cargo::rustc-link-lib=dylib=ConcreteSys");
    println!("cargo::rustc-link-lib=z");
    #[cfg(target_os = "macos")]
    {
        println!("cargo::rustc-link-search=/opt/homebrew/lib");
        println!("cargo::rustc-link-lib=curses");
        println!("cargo::rustc-link-lib=zstd");
    }
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed={}", from.display());
}

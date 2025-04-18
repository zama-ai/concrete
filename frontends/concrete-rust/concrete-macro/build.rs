use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let target_dir = out_dir.join("../../../..").canonicalize().unwrap();
    let concrete_dir = target_dir.join("concrete");
    println!(
        "cargo:rustc-env=CONCRETE_BUILD_DIR={}",
        concrete_dir.display()
    );
    #[cfg(target_os = "macos")]
    {
        println!("cargo::rustc-link-arg=-rpath");
        println!("cargo::rustc-link-arg={}", concrete_dir.display());
    }
}

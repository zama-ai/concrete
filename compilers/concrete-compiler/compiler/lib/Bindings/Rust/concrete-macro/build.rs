fn main() {
    let concrete_build_dir =
        std::env::var("DEP_CONCRETE_BUILD_DIR").expect("Failed to get concrete build dir.");
    println!(
        "cargo:rustc-env=CONCRETE_BUILD_DIR={}",
        concrete_build_dir
    );
    println!("cargo::rustc-link-arg=-rpath");
    println!("cargo::rustc-link-arg={}", concrete_build_dir);
}

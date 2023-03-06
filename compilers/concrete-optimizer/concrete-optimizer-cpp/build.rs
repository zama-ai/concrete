fn main() {
    let _build = cxx_build::bridge("src/concrete-optimizer.rs");

    println!("cargo:rerun-if-changed=src/");
}

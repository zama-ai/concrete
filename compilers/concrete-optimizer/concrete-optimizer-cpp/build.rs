fn main() {
    cxx_build::bridge("src/concrete-optimizer.rs")
        .std("c++17")
        .compile("concrete-optimizer-bridge");

    println!("cargo:rustc-link-lib=static=concrete-optimizer-bridge");
    println!("cargo:rerun-if-changed=src/");
}

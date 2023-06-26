use std::env;
use std::process::Command;

fn main() {
    println!("Build concrete-cuda");
    if env::consts::OS == "linux" {
        let output = Command::new("./get_os_name.sh").output().unwrap();
        let distribution = String::from_utf8(output.stdout).unwrap();
        if distribution != "Ubuntu\n" {
            println!(
                "cargo:warning=This Linux distribution is not officially supported. \
                Only Ubuntu is supported by concrete-cuda at this time. Build may fail\n"
            );
        }
        let dest = cmake::build("../implementation");
        println!("cargo:rustc-link-search=native={}", dest.display());
        println!("cargo:rustc-link-lib=static=concrete_cuda");
        println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu/");
        println!("cargo:rustc-link-lib=stdc++");
    } else {
        panic!("Error: platform not supported, concrete-cuda not built (only Linux is supported)");
    }
}

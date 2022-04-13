use cbindgen::Config;
use std::env;
use std::path::{Path, PathBuf};

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let package_name = env::var("CARGO_PKG_NAME").unwrap();
    let target_dir = match env::var("CARGO_TARGET_DIR") {
        Ok(target) => PathBuf::from(target),
        _ => PathBuf::from(crate_dir.clone()).join("../target"),
    };

    let header_name = format!("{}.h", package_name);
    let header_path = target_dir
        .join("include")
        .join(header_name)
        .display()
        .to_string();

    let config = Config::from_file(Path::new(&crate_dir).join("cbindgen.toml")).unwrap();

    let result = cbindgen::generate_with_config(&crate_dir, config);
    let exit_code = match result {
        Err(err) => {
            eprintln!("{}", err);
            1
        }
        Ok(content) => {
            let _changed = content.write_to_file(&header_path);
            0
        }
    };
    std::process::exit(exit_code)
}

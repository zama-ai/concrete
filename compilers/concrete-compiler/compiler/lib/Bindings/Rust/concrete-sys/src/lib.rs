#![allow(unused_imports)]

use cxx::UniquePtr;

#[cxx::bridge(namespace = "concrete_sys")]
mod ffi {

    unsafe extern "C++" {
        include!("lib.h");

        // ------------------------------------------------------------------------------- CompilationOptions
        type CompilationOptions;
        fn New() -> UniquePtr<CompilationOptions>;

        // -------------------------------------------------------------------------------------- ProgramInfo
        type ProgramInfo;

        // ------------------------------------------------------------------------------------------ Library
        type Library;
        fn getStaticLibraryPath(self: Pin<&mut Library>) -> String;
        fn getProgramInfo(self: Pin<&mut Library>) -> UniquePtr<ProgramInfo>;

        // ------------------------------------------------------------------------------------------ compile
        fn compile(
            sources: &str,
            options: &CompilationOptions,
            output_dir_path: &str,
        ) -> Result<UniquePtr<Library>>;
    }
}
pub use ffi::*;

#[cfg(test)]
mod test {
    use super::*;

    const TEST_FOLDER: &str = "/tmp/test_concrete_sys";

    #[test]
    fn test_compile() {
        let sources = "
                func.func @dec(%arg0: !FHE.eint<3>) -> !FHE.eint<3> {
                  %cst_1 = arith.constant 1 : i4
                  %1 = \"FHE.sub_eint_int\"(%arg0, %cst_1) : (!FHE.eint<3>, i4) -> !FHE.eint<3>
                  return %1: !FHE.eint<3>
                }
            ";
        let options = New();
        let _ = std::fs::remove_dir(TEST_FOLDER);
        let _ = std::fs::create_dir_all(TEST_FOLDER);
        let mut output = compile(sources, &options, TEST_FOLDER).unwrap();
        let static_library_path = output.as_mut().unwrap().getStaticLibraryPath();
        let mut archive = ar::Archive::new(std::fs::File::open(static_library_path).unwrap());
        let symbols = archive
            .symbols()
            .expect("failed to parse symbols")
            .map(|a| String::from_utf8_lossy(a).to_string())
            .collect::<Vec<_>>();
        #[cfg(target_os = "macos")]
        assert!(symbols.contains(&"_concrete_dec".to_string()));
        #[cfg(target_os = "linux")]
        assert!(symbols.contains(&"concrete_dec".to_string()));
    }
}

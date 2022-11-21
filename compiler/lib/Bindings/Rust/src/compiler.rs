//! Compiler module

use crate::mlir::ffi::*;

/// Parse the MLIR code and returns it.
///
/// The function parse the provided MLIR textual representation and returns it. It would fail with
/// an error message to stderr reporting what's bad with the parsed IR.
///
/// # Examples
/// ```
/// use concrete_compiler_rust::compiler::*;
///
/// let module_to_compile = "
///     func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
///         %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
///         return %0 : !FHE.eint<5>
///     }";
/// let result_str = round_trip(module_to_compile);
/// ```
///
pub fn round_trip(mlir_code: &str) -> String {
    unsafe {
        let engine = compilerEngineCreate();
        let mlir_code_buffer = mlir_code.as_bytes();
        let compilation_result = compilerEngineCompile(
            engine,
            MlirStringRef {
                data: mlir_code_buffer.as_ptr() as *const std::os::raw::c_char,
                length: mlir_code_buffer.len() as size_t,
            },
            CompilationTarget_ROUND_TRIP,
        );
        let module_compiled = compilationResultGetModuleString(compilation_result);
        let result_str = String::from_utf8_lossy(std::slice::from_raw_parts(
            module_compiled.data as *const u8,
            module_compiled.length as usize,
        ))
        .to_string();
        compilationResultDestroyModuleString(module_compiled);
        compilerEngineDestroy(engine);
        result_str
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_compiler_round_trip() {
        let module_to_compile = "
                func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
                    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
                    return %0 : !FHE.eint<5>
                }";
        let result_str = round_trip(module_to_compile);
        let expected_module = "module {
  func.func @main(%arg0: !FHE.eint<5>, %arg1: !FHE.eint<5>) -> !FHE.eint<5> {
    %0 = \"FHE.add_eint\"(%arg0, %arg1) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
    return %0 : !FHE.eint<5>
  }
}
";
        assert_eq!(expected_module, result_str);
    }
}

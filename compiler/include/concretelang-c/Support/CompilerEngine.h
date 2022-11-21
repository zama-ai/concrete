// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H
#define CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque type declarations. Refer to llvm-project/mlir/include/mlir-c/IR.h for
/// more info
#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
  };                                                                           \
  typedef struct name name

DEFINE_C_API_STRUCT(CompilerEngine, void);
DEFINE_C_API_STRUCT(CompilationContext, void);
DEFINE_C_API_STRUCT(CompilationResult, void);

#undef DEFINE_C_API_STRUCT

/// NULL Pointer checkers. Generate functions to check if the struct contains a
/// null pointer.
#define DEFINE_NULL_PTR_CHECKER(funcname, storage)                             \
  bool funcname(storage s) { return s.ptr == NULL; }

DEFINE_NULL_PTR_CHECKER(compilerEngineIsNull, CompilerEngine);
DEFINE_NULL_PTR_CHECKER(compilationResultIsNull, CompilationResult);

#undef DEFINE_NULL_PTR_CHECKER

/// Each struct has a creator function that allocates memory for the underlying
/// Cpp object referenced, and a destroy function that does free this allocated
/// memory.

/// ********** CompilationTarget CAPI ******************************************

enum CompilationTarget { ROUND_TRIP, OTHER };
typedef enum CompilationTarget CompilationTarget;

/// ********** CompilerEngine CAPI *********************************************

MLIR_CAPI_EXPORTED CompilerEngine compilerEngineCreate();

MLIR_CAPI_EXPORTED void compilerEngineDestroy(CompilerEngine engine);

MLIR_CAPI_EXPORTED CompilationResult compilerEngineCompile(
    CompilerEngine engine, MlirStringRef module, CompilationTarget target);

/// ********** CompilationResult CAPI ******************************************

/// Get a string reference holding the textual representation of the compiled
/// module. The returned `MlirStringRef` should be destroyed using
/// `compilationResultDestroyModuleString` to free memory.
MLIR_CAPI_EXPORTED MlirStringRef
compilationResultGetModuleString(CompilationResult result);

/// Free memory allocated for the module string.
MLIR_CAPI_EXPORTED void compilationResultDestroyModuleString(MlirStringRef str);

#ifdef __cplusplus
}
#endif

#endif // CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

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

// TODO: add a char* to the struct that can return an error message in case of
// an error (where ptr would be null). Error messages can be returned using
// llvm::toString(error.takeError()) and allocating a buffer for the message and
// copy it. The buffer can later be freed during struct destruction.
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
DEFINE_C_API_STRUCT(Library, void);
DEFINE_C_API_STRUCT(LibraryCompilationResult, void);
DEFINE_C_API_STRUCT(LibrarySupport, void);
DEFINE_C_API_STRUCT(CompilationOptions, void);
DEFINE_C_API_STRUCT(OptimizerConfig, void);

#undef DEFINE_C_API_STRUCT

/// NULL Pointer checkers. Generate functions to check if the struct contains a
/// null pointer.
#define DEFINE_NULL_PTR_CHECKER(funcname, storage)                             \
  bool funcname(storage s) { return s.ptr == NULL; }

DEFINE_NULL_PTR_CHECKER(compilerEngineIsNull, CompilerEngine);
DEFINE_NULL_PTR_CHECKER(compilationContextIsNull, CompilationContext);
DEFINE_NULL_PTR_CHECKER(compilationResultIsNull, CompilationResult);
DEFINE_NULL_PTR_CHECKER(libraryIsNull, Library);
DEFINE_NULL_PTR_CHECKER(libraryCompilationResultIsNull,
                        LibraryCompilationResult);
DEFINE_NULL_PTR_CHECKER(librarySupportIsNull, LibrarySupport);
DEFINE_NULL_PTR_CHECKER(compilationOptionsIsNull, CompilationOptions);
DEFINE_NULL_PTR_CHECKER(optimizerConfigIsNull, OptimizerConfig);

#undef DEFINE_NULL_PTR_CHECKER

/// Each struct has a creator function that allocates memory for the underlying
/// Cpp object referenced, and a destroy function that does free this allocated
/// memory.

/// ********** CompilationTarget CAPI ******************************************

enum CompilationTarget {
  ROUND_TRIP,
  FHE,
  TFHE,
  CONCRETE,
  CONCRETEWITHLOOPS,
  BCONCRETE,
  STD,
  LLVM,
  LLVM_IR,
  OPTIMIZED_LLVM_IR,
  LIBRARY
};
typedef enum CompilationTarget CompilationTarget;

/// ********** CompilationOptions CAPI *****************************************

MLIR_CAPI_EXPORTED CompilationOptions compilationOptionsCreate();

MLIR_CAPI_EXPORTED CompilationOptions compilationOptionsCreateDefault();

/// ********** OptimizerConfig CAPI ********************************************

MLIR_CAPI_EXPORTED OptimizerConfig optimizerConfigCreate();

MLIR_CAPI_EXPORTED OptimizerConfig optimizerConfigCreateDefault();

/// ********** CompilerEngine CAPI *********************************************

MLIR_CAPI_EXPORTED CompilerEngine compilerEngineCreate();

MLIR_CAPI_EXPORTED void compilerEngineDestroy(CompilerEngine engine);

MLIR_CAPI_EXPORTED CompilationResult compilerEngineCompile(
    CompilerEngine engine, MlirStringRef module, CompilationTarget target);

MLIR_CAPI_EXPORTED void
compilerEngineCompileSetOptions(CompilerEngine engine,
                                CompilationOptions options);

/// ********** CompilationResult CAPI ******************************************

/// Get a string reference holding the textual representation of the compiled
/// module. The returned `MlirStringRef` should be destroyed using
/// `compilationResultDestroyModuleString` to free memory.
MLIR_CAPI_EXPORTED MlirStringRef
compilationResultGetModuleString(CompilationResult result);

/// Free memory allocated for the module string.
MLIR_CAPI_EXPORTED void compilationResultDestroyModuleString(MlirStringRef str);

MLIR_CAPI_EXPORTED void compilationResultDestroy(CompilationResult result);

/// ********** Library CAPI ****************************************************

MLIR_CAPI_EXPORTED Library libraryCreate(MlirStringRef outputDirPath,
                                         MlirStringRef runtimeLibraryPath,
                                         bool cleanUp);

MLIR_CAPI_EXPORTED void libraryDestroy(Library lib);

/// ********** LibraryCompilationResult CAPI ***********************************

MLIR_CAPI_EXPORTED void
libraryCompilationResultDestroy(LibraryCompilationResult result);

/// ********** LibrarySupport CAPI *********************************************
MLIR_CAPI_EXPORTED LibrarySupport
librarySupportCreate(MlirStringRef outputDirPath,
                     MlirStringRef runtimeLibraryPath, bool generateSharedLib,
                     bool generateStaticLib, bool generateClientParameters,
                     bool generateCompilationFeedback, bool generateCppHeader);

MLIR_CAPI_EXPORTED LibrarySupport librarySupportCreateDefault(
    MlirStringRef outputDirPath, MlirStringRef runtimeLibraryPath) {
  return librarySupportCreate(outputDirPath, runtimeLibraryPath, true, true,
                              true, true, true);
}

MLIR_CAPI_EXPORTED LibraryCompilationResult librarySupportCompile(
    LibrarySupport support, MlirStringRef module, CompilationOptions options);

#ifdef __cplusplus
}
#endif

#endif // CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

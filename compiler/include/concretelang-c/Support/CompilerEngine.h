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

/// The CAPI should be really careful about memory allocation. Every pointer
/// returned should points to a new buffer allocated for the purpose of the
/// CAPI, and should have a respective destructor function.

/// Opaque type declarations. Inspired from
/// llvm-project/mlir/include/mlir-c/IR.h
///
/// Adds an error pointer to an allocated buffer holding the error message if
/// any.
#define DEFINE_C_API_STRUCT(name, storage)                                     \
  struct name {                                                                \
    storage *ptr;                                                              \
    char *error;                                                               \
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
DEFINE_C_API_STRUCT(ServerLambda, void);
DEFINE_C_API_STRUCT(ClientParameters, void);
DEFINE_C_API_STRUCT(KeySet, void);
DEFINE_C_API_STRUCT(KeySetCache, void);
DEFINE_C_API_STRUCT(EvaluationKeys, void);
DEFINE_C_API_STRUCT(LambdaArgument, void);
DEFINE_C_API_STRUCT(PublicArguments, void);
DEFINE_C_API_STRUCT(PublicResult, void);
DEFINE_C_API_STRUCT(CompilationFeedback, void);

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
DEFINE_NULL_PTR_CHECKER(serverLambdaIsNull, ServerLambda);
DEFINE_NULL_PTR_CHECKER(clientParametersIsNull, ClientParameters);
DEFINE_NULL_PTR_CHECKER(keySetIsNull, KeySet);
DEFINE_NULL_PTR_CHECKER(keySetCacheIsNull, KeySetCache);
DEFINE_NULL_PTR_CHECKER(evaluationKeysIsNull, EvaluationKeys);
DEFINE_NULL_PTR_CHECKER(lambdaArgumentIsNull, LambdaArgument);
DEFINE_NULL_PTR_CHECKER(publicArgumentsIsNull, PublicArguments);
DEFINE_NULL_PTR_CHECKER(publicResultIsNull, PublicResult);
DEFINE_NULL_PTR_CHECKER(compilationFeedbackIsNull, CompilationFeedback);

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

MLIR_CAPI_EXPORTED ServerLambda librarySupportLoadServerLambda(
    LibrarySupport support, LibraryCompilationResult result);

MLIR_CAPI_EXPORTED ClientParameters librarySupportLoadClientParameters(
    LibrarySupport support, LibraryCompilationResult result);

MLIR_CAPI_EXPORTED CompilationFeedback librarySupportLoadCompilationFeedback(
    LibrarySupport support, LibraryCompilationResult result);

MLIR_CAPI_EXPORTED PublicResult
librarySupportServerCall(LibrarySupport support, ServerLambda server,
                         PublicArguments args, EvaluationKeys evalKeys);

MLIR_CAPI_EXPORTED void librarySupportDestroy(LibrarySupport support);

/// ********** ServerLamda CAPI ************************************************

MLIR_CAPI_EXPORTED void serverLambdaDestroy(ServerLambda server);

/// ********** ClientParameters CAPI *******************************************

MLIR_CAPI_EXPORTED void clientParametersDestroy(ClientParameters params);

/// ********** KeySet CAPI *****************************************************

MLIR_CAPI_EXPORTED KeySet keySetGenerate(ClientParameters params,
                                         uint64_t seed_msb, uint64_t seed_lsb);

MLIR_CAPI_EXPORTED EvaluationKeys keySetGetEvaluationKeys(KeySet keySet);

MLIR_CAPI_EXPORTED void keySetDestroy(KeySet keySet);

/// ********** KeySetCache CAPI ************************************************

MLIR_CAPI_EXPORTED KeySetCache keySetCacheCreate(MlirStringRef cachePath);

MLIR_CAPI_EXPORTED KeySet
keySetCacheLoadOrGenerateKeySet(KeySetCache cache, ClientParameters params,
                                uint64_t seed_msb, uint64_t seed_lsb);

MLIR_CAPI_EXPORTED void keySetCacheDestroy(KeySetCache keySetCache);

/// ********** EvaluationKeys CAPI *********************************************

MLIR_CAPI_EXPORTED void evaluationKeysDestroy(EvaluationKeys evaluationKeys);

/// ********** LambdaArgument CAPI *********************************************

MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromScalar(uint64_t value);

MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU8(uint8_t *data,
                                                             int64_t *dims,
                                                             size_t rank);
MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU16(uint16_t *data,
                                                              int64_t *dims,
                                                              size_t rank);
MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU32(uint32_t *data,
                                                              int64_t *dims,
                                                              size_t rank);
MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU64(uint64_t *data,
                                                              int64_t *dims,
                                                              size_t rank);

MLIR_CAPI_EXPORTED bool lambdaArgumentIsScalar(LambdaArgument lambdaArg);
MLIR_CAPI_EXPORTED uint64_t lambdaArgumentGetScalar(LambdaArgument lambdaArg);

MLIR_CAPI_EXPORTED bool lambdaArgumentIsTensor(LambdaArgument lambdaArg);
MLIR_CAPI_EXPORTED bool lambdaArgumentGetTensorData(LambdaArgument lambdaArg,
                                                    uint64_t *buffer);
MLIR_CAPI_EXPORTED size_t lambdaArgumentGetTensorRank(LambdaArgument lambdaArg);
MLIR_CAPI_EXPORTED int64_t
lambdaArgumentGetTensorDataSize(LambdaArgument lambdaArg);
MLIR_CAPI_EXPORTED bool lambdaArgumentGetTensorDims(LambdaArgument lambdaArg,
                                                    int64_t *buffer);

MLIR_CAPI_EXPORTED PublicArguments
lambdaArgumentEncrypt(const LambdaArgument *lambdaArgs, size_t argNumber,
                      ClientParameters params, KeySet keySet);

MLIR_CAPI_EXPORTED void lambdaArgumentDestroy(LambdaArgument lambdaArg);

/// ********** PublicArguments CAPI ********************************************

MLIR_CAPI_EXPORTED void publicArgumentsDestroy(PublicArguments publicArgs);

/// ********** PublicResult CAPI ***********************************************

MLIR_CAPI_EXPORTED LambdaArgument publicResultDecrypt(PublicResult publicResult,
                                                      KeySet keySet);

MLIR_CAPI_EXPORTED void publicResultDestroy(PublicResult publicResult);

/// ********** CompilationFeedback CAPI ****************************************

MLIR_CAPI_EXPORTED double
compilationFeedbackGetComplexity(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED double
compilationFeedbackGetPError(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED double
compilationFeedbackGetGlobalPError(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED uint64_t
compilationFeedbackGetTotalSecretKeysSize(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED uint64_t
compilationFeedbackGetTotalBootstrapKeysSize(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED uint64_t
compilationFeedbackGetTotalKeyswitchKeysSize(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED uint64_t
compilationFeedbackGetTotalInputsSize(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED uint64_t
compilationFeedbackGetTotalOutputsSize(CompilationFeedback feedback);

MLIR_CAPI_EXPORTED void
compilationFeedbackDestroy(CompilationFeedback feedback);

#ifdef __cplusplus
}
#endif

#endif // CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

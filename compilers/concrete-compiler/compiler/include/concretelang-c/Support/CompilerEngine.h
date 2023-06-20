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
    const char *error;                                                         \
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
DEFINE_C_API_STRUCT(Encoding, void);
DEFINE_C_API_STRUCT(EncryptionGate, void);
DEFINE_C_API_STRUCT(CircuitGate, void);
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

DEFINE_NULL_PTR_CHECKER(compilerEngineIsNull, CompilerEngine)
DEFINE_NULL_PTR_CHECKER(compilationContextIsNull, CompilationContext)
DEFINE_NULL_PTR_CHECKER(compilationResultIsNull, CompilationResult)
DEFINE_NULL_PTR_CHECKER(libraryIsNull, Library)
DEFINE_NULL_PTR_CHECKER(libraryCompilationResultIsNull,
                        LibraryCompilationResult)
DEFINE_NULL_PTR_CHECKER(librarySupportIsNull, LibrarySupport)
DEFINE_NULL_PTR_CHECKER(compilationOptionsIsNull, CompilationOptions)
DEFINE_NULL_PTR_CHECKER(optimizerConfigIsNull, OptimizerConfig)
DEFINE_NULL_PTR_CHECKER(serverLambdaIsNull, ServerLambda)
DEFINE_NULL_PTR_CHECKER(circuitGateIsNull, CircuitGate)
DEFINE_NULL_PTR_CHECKER(encodingIsNull, Encoding)
DEFINE_NULL_PTR_CHECKER(encryptionGateIsNull, EncryptionGate)
DEFINE_NULL_PTR_CHECKER(clientParametersIsNull, ClientParameters)
DEFINE_NULL_PTR_CHECKER(keySetIsNull, KeySet)
DEFINE_NULL_PTR_CHECKER(keySetCacheIsNull, KeySetCache)
DEFINE_NULL_PTR_CHECKER(evaluationKeysIsNull, EvaluationKeys)
DEFINE_NULL_PTR_CHECKER(lambdaArgumentIsNull, LambdaArgument)
DEFINE_NULL_PTR_CHECKER(publicArgumentsIsNull, PublicArguments)
DEFINE_NULL_PTR_CHECKER(publicResultIsNull, PublicResult)
DEFINE_NULL_PTR_CHECKER(compilationFeedbackIsNull, CompilationFeedback)

#undef DEFINE_NULL_PTR_CHECKER

/// Each struct has a creator function that allocates memory for the underlying
/// Cpp object referenced, and a destroy function that does free this allocated
/// memory.

/// ********** Utilities *******************************************************

/// Destroy string references created by the compiler.
///
/// This is not supposed to destroy any string ref, but only the ones we have
/// allocated memory for and know how to free.
MLIR_CAPI_EXPORTED void mlirStringRefDestroy(MlirStringRef str);

MLIR_CAPI_EXPORTED bool mlirStringRefIsNull(MlirStringRef str) {
  return str.data == NULL;
}

/// ********** BufferRef CAPI **************************************************

/// A struct for binary buffers.
///
/// Contraty to MlirStringRef, it doesn't assume the pointer point to a null
/// terminated string and the data should be considered as is in binary form.
/// Useful for serialized objects.
typedef struct BufferRef {
  const char *data;
  size_t length;
  const char *error;
} BufferRef;

MLIR_CAPI_EXPORTED void bufferRefDestroy(BufferRef buffer);

MLIR_CAPI_EXPORTED bool bufferRefIsNull(BufferRef buffer) {
  return buffer.data == NULL;
}

MLIR_CAPI_EXPORTED BufferRef bufferRefCreate(const char *buffer, size_t length);

/// ********** CompilationTarget CAPI ******************************************

enum CompilationTarget {
  ROUND_TRIP,
  FHE,
  TFHE,
  PARAMETRIZED_TFHE,
  NORMALIZED_TFHE,
  BATCHED_TFHE,
  CONCRETE,
  STD,
  LLVM,
  LLVM_IR,
  OPTIMIZED_LLVM_IR,
  LIBRARY
};
typedef enum CompilationTarget CompilationTarget;

/// ********** CompilationOptions CAPI *****************************************

MLIR_CAPI_EXPORTED CompilationOptions compilationOptionsCreate(
    MlirStringRef funcName, bool autoParallelize, bool batchTFHEOps,
    bool dataflowParallelize, bool emitGPUOps, bool loopParallelize,
    bool optimizeTFHE, OptimizerConfig optimizerConfig, bool verifyDiagnostics);

MLIR_CAPI_EXPORTED CompilationOptions compilationOptionsCreateDefault();

MLIR_CAPI_EXPORTED void compilationOptionsDestroy(CompilationOptions options);

/// ********** OptimizerConfig CAPI ********************************************

MLIR_CAPI_EXPORTED OptimizerConfig
optimizerConfigCreate(bool display, double fallback_log_norm_woppbs,
                      double global_p_error, double p_error, uint64_t security,
                      bool strategy_v0, bool use_gpu_constraints,
                      uint32_t ciphertext_modulus_log, uint32_t fft_precision);

MLIR_CAPI_EXPORTED OptimizerConfig optimizerConfigCreateDefault();

MLIR_CAPI_EXPORTED void optimizerConfigDestroy(OptimizerConfig config);

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
/// `mlirStringRefDestroy` to free memory.
MLIR_CAPI_EXPORTED MlirStringRef
compilationResultGetModuleString(CompilationResult result);

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

MLIR_CAPI_EXPORTED LibraryCompilationResult
librarySupportLoadCompilationResult(LibrarySupport support);

MLIR_CAPI_EXPORTED CompilationFeedback librarySupportLoadCompilationFeedback(
    LibrarySupport support, LibraryCompilationResult result);

MLIR_CAPI_EXPORTED PublicResult
librarySupportServerCall(LibrarySupport support, ServerLambda server,
                         PublicArguments args, EvaluationKeys evalKeys);

MLIR_CAPI_EXPORTED MlirStringRef
librarySupportGetSharedLibPath(LibrarySupport support);

MLIR_CAPI_EXPORTED MlirStringRef
librarySupportGetProgramInfoPath(LibrarySupport support);

MLIR_CAPI_EXPORTED void librarySupportDestroy(LibrarySupport support);

/// ********** ServerLamda CAPI ************************************************

MLIR_CAPI_EXPORTED void serverLambdaDestroy(ServerLambda server);

/// ********** ClientParameters CAPI *******************************************

MLIR_CAPI_EXPORTED BufferRef clientParametersSerialize(ClientParameters params);

MLIR_CAPI_EXPORTED ClientParameters
clientParametersUnserialize(BufferRef buffer);

MLIR_CAPI_EXPORTED ClientParameters
clientParametersCopy(ClientParameters params);

MLIR_CAPI_EXPORTED void clientParametersDestroy(ClientParameters params);

/// Returns the number of output circuit gates
MLIR_CAPI_EXPORTED size_t clientParametersOutputsSize(ClientParameters params);

/// Returns the number of input circuit gates
MLIR_CAPI_EXPORTED size_t clientParametersInputsSize(ClientParameters params);

/// Returns the output circuit gate corresponding to the index
///
/// - `index` must be valid.
MLIR_CAPI_EXPORTED CircuitGate
clientParametersOutputCircuitGate(ClientParameters params, size_t index);

/// Returns the input circuit gate corresponding to the index
///
/// - `index` must be valid.
MLIR_CAPI_EXPORTED CircuitGate
clientParametersInputCircuitGate(ClientParameters params, size_t index);

/// Returns the EncryptionGate of the circuit gate.
///
/// - The returned gate will be null if the gate does not represent encrypted
/// data
MLIR_CAPI_EXPORTED EncryptionGate
circuitGateEncryptionGate(CircuitGate circuit_gate);

/// Returns the variance of the encryption gate
MLIR_CAPI_EXPORTED double
encryptionGateVariance(EncryptionGate encryption_gate);

/// Returns the Encoding of the encryption gate.
MLIR_CAPI_EXPORTED Encoding
encryptionGateEncoding(EncryptionGate encryption_gate);

/// Returns the precision (bit width) of the encoding
MLIR_CAPI_EXPORTED uint64_t encodingPrecision(Encoding encoding);

MLIR_CAPI_EXPORTED void circuitGateDestroy(CircuitGate gate);
MLIR_CAPI_EXPORTED void encryptionGateDestroy(EncryptionGate gate);
MLIR_CAPI_EXPORTED void encodingDestroy(Encoding encoding);

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

MLIR_CAPI_EXPORTED BufferRef evaluationKeysSerialize(EvaluationKeys keys);

MLIR_CAPI_EXPORTED EvaluationKeys evaluationKeysUnserialize(BufferRef buffer);

MLIR_CAPI_EXPORTED void evaluationKeysDestroy(EvaluationKeys evaluationKeys);

/// ********** LambdaArgument CAPI *********************************************

MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromScalar(uint64_t value);

MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU8(
    const uint8_t *data, const int64_t *dims, size_t rank);
MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU16(
    const uint16_t *data, const int64_t *dims, size_t rank);
MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU32(
    const uint32_t *data, const int64_t *dims, size_t rank);
MLIR_CAPI_EXPORTED LambdaArgument lambdaArgumentFromTensorU64(
    const uint64_t *data, const int64_t *dims, size_t rank);

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

MLIR_CAPI_EXPORTED BufferRef publicArgumentsSerialize(PublicArguments args);

MLIR_CAPI_EXPORTED PublicArguments
publicArgumentsUnserialize(BufferRef buffer, ClientParameters params);

MLIR_CAPI_EXPORTED void publicArgumentsDestroy(PublicArguments publicArgs);

/// ********** PublicResult CAPI ***********************************************

MLIR_CAPI_EXPORTED LambdaArgument publicResultDecrypt(PublicResult publicResult,
                                                      KeySet keySet);

MLIR_CAPI_EXPORTED BufferRef publicResultSerialize(PublicResult result);

MLIR_CAPI_EXPORTED PublicResult
publicResultUnserialize(BufferRef buffer, ClientParameters params);

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

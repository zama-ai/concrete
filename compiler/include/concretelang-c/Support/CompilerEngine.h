// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H
#define CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/Jit.h"
#include "concretelang/Support/LibrarySupport.h"
#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper of the mlir::concretelang::LambdaArgument
struct lambdaArgument {
  std::shared_ptr<mlir::concretelang::LambdaArgument> ptr;
};
typedef struct lambdaArgument lambdaArgument;

// Hold a list of lambdaArgument to represent execution arguments
struct executionArguments {
  lambdaArgument *data;
  size_t size;
};
typedef struct executionArguments executionArguments;

// JIT Support bindings ///////////////////////////////////////////////////////

struct JITSupport_C {
  mlir::concretelang::JITSupport support;
};
typedef struct JITSupport_C JITSupport_C;

MLIR_CAPI_EXPORTED JITSupport_C jit_support(std::string runtimeLibPath);

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::JitCompilationResult>
jit_compile(JITSupport_C support, const char *module,
            mlir::concretelang::CompilationOptions options);

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
jit_load_client_parameters(JITSupport_C support,
                           mlir::concretelang::JitCompilationResult &);

MLIR_CAPI_EXPORTED std::shared_ptr<mlir::concretelang::JITLambda>
jit_load_server_lambda(JITSupport_C support,
                       mlir::concretelang::JitCompilationResult &);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
jit_server_call(JITSupport_C support, mlir::concretelang::JITLambda &lambda,
                concretelang::clientlib::PublicArguments &args,
                concretelang::clientlib::EvaluationKeys &evaluationKeys);

// Library Support bindings ///////////////////////////////////////////////////

struct LibrarySupport_C {
  mlir::concretelang::LibrarySupport support;
};
typedef struct LibrarySupport_C LibrarySupport_C;

MLIR_CAPI_EXPORTED LibrarySupport_C
library_support(const char *outputPath, const char *runtimeLibraryPath,
                bool generateSharedLib, bool generateStaticLib,
                bool generateClientParameters, bool generateCppHeader);

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibrarySupport_C support, const char *module,
                mlir::concretelang::CompilationOptions options);

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
library_load_client_parameters(LibrarySupport_C support,
                               mlir::concretelang::LibraryCompilationResult &);

MLIR_CAPI_EXPORTED concretelang::serverlib::ServerLambda
library_load_server_lambda(LibrarySupport_C support,
                           mlir::concretelang::LibraryCompilationResult &);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_server_call(LibrarySupport_C support,
                    concretelang::serverlib::ServerLambda lambda,
                    concretelang::clientlib::PublicArguments &args,
                    concretelang::clientlib::EvaluationKeys &evaluationKeys);

MLIR_CAPI_EXPORTED std::string
library_get_shared_lib_path(LibrarySupport_C support);

MLIR_CAPI_EXPORTED std::string
library_get_client_parameters_path(LibrarySupport_C support);

// Client Support bindings ///////////////////////////////////////////////////

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
key_set(concretelang::clientlib::ClientParameters clientParameters,
        llvm::Optional<concretelang::clientlib::KeySetCache> cache);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
encrypt_arguments(concretelang::clientlib::ClientParameters clientParameters,
                  concretelang::clientlib::KeySet &keySet,
                  llvm::ArrayRef<mlir::concretelang::LambdaArgument *> args);

MLIR_CAPI_EXPORTED lambdaArgument
decrypt_result(concretelang::clientlib::KeySet &keySet,
               concretelang::clientlib::PublicResult &publicResult);

// Serialization ////////////////////////////////////////////////////////////

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
clientParametersUnserialize(const std::string &json);

MLIR_CAPI_EXPORTED std::string
clientParametersSerialize(mlir::concretelang::ClientParameters &params);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
publicArgumentsUnserialize(
    mlir::concretelang::ClientParameters &clientParameters,
    const std::string &buffer);

MLIR_CAPI_EXPORTED std::string publicArgumentsSerialize(
    concretelang::clientlib::PublicArguments &publicArguments);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
publicResultUnserialize(mlir::concretelang::ClientParameters &clientParameters,
                        const std::string &buffer);

MLIR_CAPI_EXPORTED std::string
publicResultSerialize(concretelang::clientlib::PublicResult &publicResult);

MLIR_CAPI_EXPORTED concretelang::clientlib::EvaluationKeys
evaluationKeysUnserialize(const std::string &buffer);

MLIR_CAPI_EXPORTED std::string evaluationKeysSerialize(
    concretelang::clientlib::EvaluationKeys &evaluationKeys);

// Parse then print a textual representation of an MLIR module
MLIR_CAPI_EXPORTED std::string roundTrip(const char *module);

// Terminate parallelization
MLIR_CAPI_EXPORTED void terminateParallelization();

// Create a lambdaArgument from a tensor of different data types
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU8(
    std::vector<uint8_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU16(
    std::vector<uint16_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU32(
    std::vector<uint32_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU64(
    std::vector<uint64_t> data, std::vector<int64_t> dimensions);
// Create a lambdaArgument from a scalar
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromScalar(uint64_t scalar);
// Check if a lambdaArgument holds a tensor
MLIR_CAPI_EXPORTED bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg);
// Get tensor data from lambdaArgument
MLIR_CAPI_EXPORTED std::vector<uint64_t>
lambdaArgumentGetTensorData(lambdaArgument &lambda_arg);
// Get tensor dimensions from lambdaArgument
MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg);
// Check if a lambdaArgument holds a scalar
MLIR_CAPI_EXPORTED bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg);
// Get scalar value from lambdaArgument
MLIR_CAPI_EXPORTED uint64_t lambdaArgumentGetScalar(lambdaArgument &lambda_arg);

// Compile the textual representation of MLIR modules to a library.
MLIR_CAPI_EXPORTED std::string library(std::string libraryPath,
                                       std::vector<std::string> modules);

#ifdef __cplusplus
}
#endif

#endif // CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

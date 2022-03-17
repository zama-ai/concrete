// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H
#define CONCRETELANG_C_SUPPORT_COMPILER_ENGINE_H

#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Jit.h"
#include "concretelang/Support/JitCompilerEngine.h"
#include "concretelang/Support/JitLambdaSupport.h"
#include "concretelang/Support/LibraryLambdaSupport.h"
#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper of the mlir::concretelang::JitCompilerEngine::Lambda
struct lambda {
  mlir::concretelang::JitCompilerEngine::Lambda *ptr;
};
typedef struct lambda lambda;

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

struct JITLambdaSupport_C {
  mlir::concretelang::JitLambdaSupport support;
};
typedef struct JITLambdaSupport_C JITLambdaSupport_C;

MLIR_CAPI_EXPORTED JITLambdaSupport_C
jit_lambda_support(const char *runtimeLibPath);

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::JitCompilationResult>
jit_compile(JITLambdaSupport_C support, const char *module,
            mlir::concretelang::CompilationOptions options);

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
jit_load_client_parameters(JITLambdaSupport_C support,
                           mlir::concretelang::JitCompilationResult &);

MLIR_CAPI_EXPORTED mlir::concretelang::JITLambda *
jit_load_server_lambda(JITLambdaSupport_C support,
                       mlir::concretelang::JitCompilationResult &);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
jit_server_call(JITLambdaSupport_C support,
                mlir::concretelang::JITLambda *lambda,
                concretelang::clientlib::PublicArguments &args);

// Library Support bindings ///////////////////////////////////////////////////

struct LibraryLambdaSupport_C {
  mlir::concretelang::LibraryLambdaSupport support;
};
typedef struct LibraryLambdaSupport_C LibraryLambdaSupport_C;

MLIR_CAPI_EXPORTED LibraryLambdaSupport_C
library_lambda_support(const char *outputPath);

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibraryLambdaSupport_C support, const char *module,
                mlir::concretelang::CompilationOptions options);

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
library_load_client_parameters(LibraryLambdaSupport_C support,
                               mlir::concretelang::LibraryCompilationResult &);

MLIR_CAPI_EXPORTED concretelang::serverlib::ServerLambda
library_load_server_lambda(LibraryLambdaSupport_C support,
                           mlir::concretelang::LibraryCompilationResult &);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_server_call(LibraryLambdaSupport_C support,
                    concretelang::serverlib::ServerLambda lambda,
                    concretelang::clientlib::PublicArguments &args);

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

// Build lambda from a textual representation of an MLIR module
// The lambda will have `funcName` as entrypoint, and use runtimeLibPath (if
// not null) as a shared library during compilation, a path to activate the
// use a cache for encryption keys for test purpose (unsecure).
MLIR_CAPI_EXPORTED mlir::concretelang::JitCompilerEngine::Lambda
buildLambda(const char *module, const char *funcName,
            const char *runtimeLibPath, const char *keySetCachePath,
            bool autoParallelize, bool loopParallelize, bool dfParallelize);

// Parse then print a textual representation of an MLIR module
MLIR_CAPI_EXPORTED std::string roundTrip(const char *module);

// Execute the lambda with executionArguments and get the result as
// lambdaArgument
MLIR_CAPI_EXPORTED lambdaArgument invokeLambda(lambda l,
                                               executionArguments args);

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

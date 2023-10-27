// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_BINDINGS_PYTHON_COMPILER_ENGINE_H
#define CONCRETELANG_BINDINGS_PYTHON_COMPILER_ENGINE_H

#include "concretelang/Common/Compat.h"
#include "concretelang/Support/CompilerEngine.h"
#include "mlir-c/IR.h"

/// MLIR_CAPI_EXPORTED is used here throughout the API, because of the way the
/// python extension is built using MLIR cmake functions, which will cause
/// undefined symbols during runtime if those aren't present.

/// Wrapper of the mlir::concretelang::LambdaArgument
struct lambdaArgument {
  std::shared_ptr<mlir::concretelang::LambdaArgument> ptr;
};
typedef struct lambdaArgument lambdaArgument;

/// Hold a list of lambdaArgument to represent execution arguments
struct executionArguments {
  lambdaArgument *data;
  size_t size;
};
typedef struct executionArguments executionArguments;

// Library Support bindings ///////////////////////////////////////////////////

struct LibrarySupport_Py {
  mlir::concretelang::LibrarySupport support;
};
typedef struct LibrarySupport_Py LibrarySupport_Py;

MLIR_CAPI_EXPORTED LibrarySupport_Py
library_support(const char *outputPath, const char *runtimeLibraryPath,
                bool generateSharedLib, bool generateStaticLib,
                bool generateClientParameters, bool generateCompilationFeedback,
                bool generateCppHeader);

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile_module(
    LibrarySupport_Py support, mlir::ModuleOp module,
    mlir::concretelang::CompilationOptions options,
    std::shared_ptr<mlir::concretelang::CompilationContext> cctx);

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibrarySupport_Py support, const char *module,
                mlir::concretelang::CompilationOptions options);

MLIR_CAPI_EXPORTED concretelang::clientlib::ClientParameters
library_load_client_parameters(LibrarySupport_Py support,
                               mlir::concretelang::LibraryCompilationResult &);

MLIR_CAPI_EXPORTED mlir::concretelang::CompilationFeedback
library_load_compilation_feedback(
    LibrarySupport_Py support, mlir::concretelang::LibraryCompilationResult &);

MLIR_CAPI_EXPORTED concretelang::serverlib::ServerLambda
library_load_server_lambda(LibrarySupport_Py support,
                           mlir::concretelang::LibraryCompilationResult &,
                           bool useSimulation);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_server_call(LibrarySupport_Py support,
                    concretelang::serverlib::ServerLambda lambda,
                    concretelang::clientlib::PublicArguments &args,
                    concretelang::clientlib::EvaluationKeys &evaluationKeys);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_simulate(LibrarySupport_Py support,
                 concretelang::serverlib::ServerLambda lambda,
                 concretelang::clientlib::PublicArguments &args);

MLIR_CAPI_EXPORTED std::string
library_get_shared_lib_path(LibrarySupport_Py support);

MLIR_CAPI_EXPORTED std::string
library_get_program_info_path(LibrarySupport_Py support);

// Client Support bindings ///////////////////////////////////////////////////

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
key_set(concretelang::clientlib::ClientParameters clientParameters,
        std::optional<concretelang::clientlib::KeySetCache> cache,
        uint64_t seedMsb, uint64_t seedLsb);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
encrypt_arguments(concretelang::clientlib::ClientParameters clientParameters,
                  concretelang::clientlib::KeySet &keySet,
                  llvm::ArrayRef<mlir::concretelang::LambdaArgument *> args);

MLIR_CAPI_EXPORTED lambdaArgument
decrypt_result(concretelang::clientlib::ClientParameters clientParameters,
               concretelang::clientlib::KeySet &keySet,
               concretelang::clientlib::PublicResult &publicResult);

// Serialization ////////////////////////////////////////////////////////////

MLIR_CAPI_EXPORTED concretelang::clientlib::ClientParameters
clientParametersUnserialize(const std::string &json);

MLIR_CAPI_EXPORTED std::string
clientParametersSerialize(concretelang::clientlib::ClientParameters &params);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
publicArgumentsUnserialize(
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &buffer);

MLIR_CAPI_EXPORTED std::string publicArgumentsSerialize(
    concretelang::clientlib::PublicArguments &publicArguments);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
publicResultUnserialize(
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &buffer);

MLIR_CAPI_EXPORTED std::string
publicResultSerialize(concretelang::clientlib::PublicResult &publicResult);

MLIR_CAPI_EXPORTED concretelang::clientlib::EvaluationKeys
evaluationKeysUnserialize(const std::string &buffer);

MLIR_CAPI_EXPORTED std::string evaluationKeysSerialize(
    concretelang::clientlib::EvaluationKeys &evaluationKeys);

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
keySetUnserialize(const std::string &buffer);

MLIR_CAPI_EXPORTED std::string
keySetSerialize(concretelang::clientlib::KeySet &keySet);

MLIR_CAPI_EXPORTED concretelang::clientlib::SharedScalarOrTensorData
valueUnserialize(const std::string &buffer);

MLIR_CAPI_EXPORTED std::string
valueSerialize(const concretelang::clientlib::SharedScalarOrTensorData &value);

MLIR_CAPI_EXPORTED concretelang::clientlib::ValueExporter createValueExporter(
    concretelang::clientlib::KeySet &keySet,
    concretelang::clientlib::ClientParameters &clientParameters);

MLIR_CAPI_EXPORTED concretelang::clientlib::SimulatedValueExporter
createSimulatedValueExporter(
    concretelang::clientlib::ClientParameters &clientParameters);

MLIR_CAPI_EXPORTED concretelang::clientlib::ValueDecrypter createValueDecrypter(
    concretelang::clientlib::KeySet &keySet,
    concretelang::clientlib::ClientParameters &clientParameters);

MLIR_CAPI_EXPORTED concretelang::clientlib::SimulatedValueDecrypter
createSimulatedValueDecrypter(
    concretelang::clientlib::ClientParameters &clientParameters);

/// Parse then print a textual representation of an MLIR module
MLIR_CAPI_EXPORTED std::string roundTrip(const char *module);

/// Terminate/Init dataflow parallelization
MLIR_CAPI_EXPORTED void terminateDataflowParallelization();
MLIR_CAPI_EXPORTED void initDataflowParallelization();

/// Create a lambdaArgument from a tensor of different data types
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU8(
    std::vector<uint8_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU16(
    std::vector<uint16_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU32(
    std::vector<uint32_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU64(
    std::vector<uint64_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI8(
    std::vector<int8_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI16(
    std::vector<int16_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI32(
    std::vector<int32_t> data, std::vector<int64_t> dimensions);
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI64(
    std::vector<int64_t> data, std::vector<int64_t> dimensions);
/// Create a lambdaArgument from a scalar
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromScalar(uint64_t scalar);
MLIR_CAPI_EXPORTED lambdaArgument
lambdaArgumentFromSignedScalar(int64_t scalar);
/// Check if a lambdaArgument holds a tensor
MLIR_CAPI_EXPORTED bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg);
/// Get tensor data from lambdaArgument
MLIR_CAPI_EXPORTED std::vector<uint64_t>
lambdaArgumentGetTensorData(lambdaArgument &lambda_arg);
MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetSignedTensorData(lambdaArgument &lambda_arg);
/// Get tensor dimensions from lambdaArgument
MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg);
/// Check if a lambdaArgument holds a scalar
MLIR_CAPI_EXPORTED bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg);
/// Check if a lambdaArgument holds a signed value
MLIR_CAPI_EXPORTED bool lambdaArgumentIsSigned(lambdaArgument &lambda_arg);
/// Get scalar value from lambdaArgument
MLIR_CAPI_EXPORTED uint64_t lambdaArgumentGetScalar(lambdaArgument &lambda_arg);
MLIR_CAPI_EXPORTED int64_t
lambdaArgumentGetSignedScalar(lambdaArgument &lambda_arg);

/// Compile the textual representation of MLIR modules to a library.
MLIR_CAPI_EXPORTED std::string library(std::string libraryPath,
                                       std::vector<std::string> modules);

#endif // CONCRETELANG_BINDINGS_PYTHON_COMPILER_ENGINE_H

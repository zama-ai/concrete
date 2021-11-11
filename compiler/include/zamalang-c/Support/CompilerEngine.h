#ifndef ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H
#define ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/JitCompilerEngine.h"

#ifdef __cplusplus
extern "C" {
#endif

// C wrapper of the mlir::zamalang::JitCompilerEngine::Lambda
struct lambda {
  mlir::zamalang::JitCompilerEngine::Lambda *ptr;
};
typedef struct lambda lambda;

// C wrapper of the mlir::zamalang::LambdaArgument
struct lambdaArgument {
  std::shared_ptr<mlir::zamalang::LambdaArgument> ptr;
};
typedef struct lambdaArgument lambdaArgument;

// Hold a list of lambdaArgument to represent execution arguments
struct executionArguments {
  lambdaArgument *data;
  size_t size;
};
typedef struct executionArguments executionArguments;

// Build lambda from a textual representation of an MLIR module
// The lambda will have `funcName` as entrypoint, and use runtimeLibPath (if not
// null) as a shared library during compilation
MLIR_CAPI_EXPORTED mlir::zamalang::JitCompilerEngine::Lambda
buildLambda(const char *module, const char *funcName,
            const char *runtimeLibPath);

// Parse then print a textual representation of an MLIR module
MLIR_CAPI_EXPORTED std::string roundTrip(const char *module);

// Execute the lambda with executionArguments and get the result as
// lambdaArgument
MLIR_CAPI_EXPORTED lambdaArgument invokeLambda(lambda l,
                                               executionArguments args);

// Create a lambdaArgument from a tensor
MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensor(
    std::vector<uint8_t> data, std::vector<int64_t> dimensions);
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

#endif // ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

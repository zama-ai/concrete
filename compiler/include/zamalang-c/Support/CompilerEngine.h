#ifndef ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H
#define ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/ExecutionArgument.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/JitCompilerEngine.h"

#ifdef __cplusplus
extern "C" {
#endif

struct lambda {
  mlir::zamalang::JitCompilerEngine::Lambda *ptr;
};
typedef struct lambda lambda;

struct executionArguments {
  mlir::zamalang::ExecutionArgument *data;
  size_t size;
};
typedef struct executionArguments exectuionArguments;

struct lambdaArgument {
  std::unique_ptr<mlir::zamalang::LambdaArgument> ptr;
};
typedef struct lambdaArgument lambdaArgument;

MLIR_CAPI_EXPORTED mlir::zamalang::JitCompilerEngine::Lambda
buildLambda(const char *module, const char *funcName);

MLIR_CAPI_EXPORTED lambdaArgument invokeLambda(lambda l,
                                               executionArguments args);

MLIR_CAPI_EXPORTED std::string roundTrip(const char *module);

MLIR_CAPI_EXPORTED bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg);
MLIR_CAPI_EXPORTED std::vector<uint64_t>
lambdaArgumentGetTensorData(lambdaArgument &lambda_arg);
MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg);
MLIR_CAPI_EXPORTED bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg);
MLIR_CAPI_EXPORTED uint64_t lambdaArgumentGetScalar(lambdaArgument &lambda_arg);

#ifdef __cplusplus
}
#endif

#endif // ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

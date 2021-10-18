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

MLIR_CAPI_EXPORTED mlir::zamalang::JitCompilerEngine::Lambda
buildLambda(const char *module, const char *funcName);

MLIR_CAPI_EXPORTED uint64_t invokeLambda(lambda l, executionArguments args);

MLIR_CAPI_EXPORTED std::string roundTrip(const char *module);

#ifdef __cplusplus
}
#endif

#endif // ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

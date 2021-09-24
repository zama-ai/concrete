#ifndef ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H
#define ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

#include "mlir-c/IR.h"
#include "mlir-c/Registration.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/ExecutionArgument.h"

#ifdef __cplusplus
extern "C" {
#endif

struct compilerEngine {
  mlir::zamalang::CompilerEngine *ptr;
};
typedef struct compilerEngine compilerEngine;

struct executionArguments {
  mlir::zamalang::ExecutionArgument *data;
  size_t size;
};
typedef struct executionArguments exectuionArguments;

// Compile an MLIR module
MLIR_CAPI_EXPORTED void compilerEngineCompile(compilerEngine engine,
                                              const char *module);

// Run the compiled module
MLIR_CAPI_EXPORTED uint64_t compilerEngineRun(compilerEngine e,
                                              executionArguments args);

#ifdef __cplusplus
}
#endif

#endif // ZAMALANG_C_SUPPORT_COMPILER_ENGINE_H

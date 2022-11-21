// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CAPI_WRAPPERS_H
#define CONCRETELANG_CAPI_WRAPPERS_H

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/Support/CompilerEngine.h"
#include "mlir/CAPI/Wrap.h"

DEFINE_C_API_PTR_METHODS(CompilerEngine, mlir::concretelang::CompilerEngine)
DEFINE_C_API_PTR_METHODS(CompilationResult,
                         mlir::concretelang::CompilerEngine::CompilationResult)

#endif

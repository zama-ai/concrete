// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CAPI_WRAPPERS_H
#define CONCRETELANG_CAPI_WRAPPERS_H

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/LibrarySupport.h"
#include "mlir/CAPI/Wrap.h"

DEFINE_C_API_PTR_METHODS(CompilerEngine, mlir::concretelang::CompilerEngine)
DEFINE_C_API_PTR_METHODS(CompilationContext,
                         mlir::concretelang::CompilationContext)
DEFINE_C_API_PTR_METHODS(CompilationResult,
                         mlir::concretelang::CompilerEngine::CompilationResult)
DEFINE_C_API_PTR_METHODS(Library, mlir::concretelang::CompilerEngine::Library)
DEFINE_C_API_PTR_METHODS(LibraryCompilationResult,
                         mlir::concretelang::LibraryCompilationResult)
DEFINE_C_API_PTR_METHODS(LibrarySupport, mlir::concretelang::LibrarySupport)
DEFINE_C_API_PTR_METHODS(CompilationOptions,
                         mlir::concretelang::CompilationOptions)
DEFINE_C_API_PTR_METHODS(OptimizerConfig, mlir::concretelang::optimizer::Config)
DEFINE_C_API_PTR_METHODS(ServerLambda,
                         mlir::concretelang::serverlib::ServerLambda)

#endif

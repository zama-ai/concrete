// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CAPI_WRAPPERS_H
#define CONCRETELANG_CAPI_WRAPPERS_H

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/LibrarySupport.h"

/// Add a mechanism to go from Cpp objects to C-struct, with the ability to
/// represent errors. Also the other way around.
#define DEFINE_C_API_PTR_METHODS_WITH_ERROR(name, cpptype)                     \
  static inline name wrap(cpptype *cpp) { return name{cpp, (char *)NULL}; }    \
  static inline name wrap(cpptype *cpp, std::string errorStr) {                \
    char *error = new char[errorStr.size()];                                   \
    strcpy(error, errorStr.c_str());                                           \
    return name{(cpptype *)NULL, error};                                       \
  }                                                                            \
  static inline cpptype *unwrap(name c) {                                      \
    return static_cast<cpptype *>(c.ptr);                                      \
  }                                                                            \
  static inline const char *getErrorPtr(name c) { return c.error; }

DEFINE_C_API_PTR_METHODS_WITH_ERROR(CompilerEngine,
                                    mlir::concretelang::CompilerEngine)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(CompilationContext,
                                    mlir::concretelang::CompilationContext)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(
    CompilationResult, mlir::concretelang::CompilerEngine::CompilationResult)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(Library,
                                    mlir::concretelang::CompilerEngine::Library)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(
    LibraryCompilationResult, mlir::concretelang::LibraryCompilationResult)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(LibrarySupport,
                                    mlir::concretelang::LibrarySupport)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(CompilationOptions,
                                    mlir::concretelang::CompilationOptions)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(OptimizerConfig,
                                    mlir::concretelang::optimizer::Config)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(ServerLambda,
                                    mlir::concretelang::serverlib::ServerLambda)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(
    ClientParameters, mlir::concretelang::clientlib::ClientParameters)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(KeySet,
                                    mlir::concretelang::clientlib::KeySet)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(KeySetCache,
                                    mlir::concretelang::clientlib::KeySetCache)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(
    EvaluationKeys, mlir::concretelang::clientlib::EvaluationKeys)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(LambdaArgument,
                                    mlir::concretelang::LambdaArgument)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(
    PublicArguments, mlir::concretelang::clientlib::PublicArguments)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(PublicResult,
                                    mlir::concretelang::clientlib::PublicResult)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(CompilationFeedback,
                                    mlir::concretelang::CompilationFeedback)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(Encoding,
                                    mlir::concretelang::clientlib::Encoding)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(
    EncryptionGate, mlir::concretelang::clientlib::EncryptionGate)
DEFINE_C_API_PTR_METHODS_WITH_ERROR(CircuitGate,
                                    mlir::concretelang::clientlib::CircuitGate)

#undef DEFINE_C_API_PTR_METHODS_WITH_ERROR

#endif

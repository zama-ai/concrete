// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/CAPI/Wrappers.h"
#include "concretelang/Support/CompilerEngine.h"
#include "mlir/IR/Diagnostics.h"

/// CompilerEngine CAPI

CompilerEngine compilerEngineCreate() {
  auto *engine = new mlir::concretelang::CompilerEngine(
      mlir::concretelang::CompilationContext::createShared());
  return wrap(engine);
}

void compilerEngineDestroy(CompilerEngine engine) { delete unwrap(engine); }

CompilationResult compilerEngineCompile(CompilerEngine engine,
                                        MlirStringRef module,
                                        CompilationTarget target) {
  std::string module_str(module.data, module.length);
  if (target == ROUND_TRIP) {
    auto retOrError = unwrap(engine)->compile(
        module_str, mlir::concretelang::CompilerEngine::Target::ROUND_TRIP);
    if (!retOrError) {
      // TODO: access the MlirContext
      // mlir::emitError(mlir::UnknownLoc::get(unwrap(engine)) << "azeza";
      return wrap(
          (mlir::concretelang::CompilerEngine::CompilationResult *)nullptr);
    }
    return wrap(new mlir::concretelang::CompilerEngine::CompilationResult(
        std::move(retOrError.get())));
  }
  return wrap((mlir::concretelang::CompilerEngine::CompilationResult *)nullptr);
}

/// CompilationResult CAPI
void compilationResultDestroy(CompilationResult result) {
  delete unwrap(result);
}

MlirStringRef compilationResultGetModuleString(CompilationResult result) {
  // print the module into a string
  std::string moduleString;
  llvm::raw_string_ostream os(moduleString);
  unwrap(result)->mlirModuleRef->get().print(os);
  // allocate buffer and copy module string
  char *buffer = new char[moduleString.length() + 1];
  strcpy(buffer, moduleString.c_str());
  return mlirStringRefCreate(buffer, moduleString.length());
}

void compilationResultDestroyModuleString(MlirStringRef str) {
  delete str.data;
}

#ifndef ZAMALANG_SUPPORT_COMPILER_ENGINE_H
#define ZAMALANG_SUPPORT_COMPILER_ENGINE_H

#include "Jit.h"

namespace mlir {
namespace zamalang {

/// CompilerEngine is an tools that provides tools to implements the compilation
/// flow and manage the compilation flow state.
class CompilerEngine {
public:
  CompilerEngine() {
    context = new mlir::MLIRContext();
    loadDialects();
  }
  ~CompilerEngine() {
    if (context != nullptr)
      delete context;
  }

  // Compile an mlir programs from it's textual representation.
  llvm::Error compile(
      std::string mlirStr,
      llvm::Optional<mlir::zamalang::V0FHEConstraint> overrideConstraints = {});

  // Build the jit lambda argument.
  llvm::Expected<std::unique_ptr<JITLambda::Argument>> buildArgument();

  // Call the compiled function with and argument object.
  llvm::Error invoke(JITLambda::Argument &arg);

  // Call the compiled function with a list of integer arguments.
  llvm::Expected<uint64_t> run(std::vector<uint64_t> args);

  // Get a printable representation of the compiled module
  std::string getCompiledModule();

private:
  // Load the necessary dialects into the engine's context
  void loadDialects();

  mlir::OwningModuleRef module_ref;
  mlir::MLIRContext *context;
  std::unique_ptr<mlir::zamalang::KeySet> keySet;
};
} // namespace zamalang
} // namespace mlir

#endif

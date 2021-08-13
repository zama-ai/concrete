#ifndef ZAMALANG_SUPPORT_COMPILER_ENGINE_H
#define ZAMALANG_SUPPORT_COMPILER_ENGINE_H

#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"
#include "zamalang/Support/CompilerTools.h"
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <string>

namespace mlir {
namespace zamalang {
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

  // Compile an MLIR input
  llvm::Expected<mlir::LogicalResult> compileFHE(std::string mlir_input);

  // Run the compiled module
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
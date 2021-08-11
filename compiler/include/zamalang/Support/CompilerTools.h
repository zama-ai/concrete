#ifndef ZAMALANG_SUPPORT_COMPILERTOOLS_H_
#define ZAMALANG_SUPPORT_COMPILERTOOLS_H_

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Pass/PassManager.h>

#include "zamalang/Support/ClientParameters.h"
#include "zamalang/Support/KeySet.h"
#include "zamalang/Support/V0Parameters.h"

namespace mlir {
namespace zamalang {

/// For the v0 we compute a global constraint, this is defined here as the
/// high-level verification pass is not yet implemented.
struct FHECircuitConstraint {
  size_t norm2;
  size_t p;
};

class CompilerTools {
public:
  /// lowerHLFHEToMlirLLVMDialect run all passes to lower FHE dialects to mlir
  /// lowerable to llvm dialect.
  /// The given module MLIR operation would be modified and the constraints set.
  static mlir::LogicalResult lowerHLFHEToMlirStdsDialect(
      mlir::MLIRContext &context, mlir::Operation *module,
      FHECircuitConstraint &constraint, V0Parameter &v0Parameter,
      llvm::function_ref<bool(std::string)> enablePass = [](std::string pass) {
        return true;
      });

  /// lowerMlirStdsDialectToMlirLLVMDialect run all passes to lower MLIR
  /// dialects to MLIR LLVM dialect. The given module MLIR operation would be
  /// modified.
  static mlir::LogicalResult lowerMlirStdsDialectToMlirLLVMDialect(
      mlir::MLIRContext &context, mlir::Operation *module,
      llvm::function_ref<bool(std::string)> enablePass = [](std::string pass) {
        return true;
      });

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  toLLVMModule(llvm::LLVMContext &llvmContext, mlir::ModuleOp &module,
               llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline);
};

/// JITLambda is a tool to JIT compile an mlir module and to invoke a function
/// of the module.
class JITLambda {
public:
  class Argument {
  public:
    Argument(KeySet &keySet);
    ~Argument();

    // Create lambda Argument that use the given KeySet to perform encryption
    // and decryption operations.
    static llvm::Expected<std::unique_ptr<Argument>> create(KeySet &keySet);

    // Set the argument at the given pos as a uint64_t.
    llvm::Error setArg(size_t pos, uint64_t arg);

    // Get the result at the given pos as an uint64_t.
    llvm::Error getResult(size_t pos, uint64_t &res);

  private:
    friend JITLambda;
    std::vector<void *> rawArg;
    std::vector<void *> inputs;
    std::vector<void *> results;
    KeySet &keySet;
  };
  JITLambda(mlir::LLVM::LLVMFunctionType type, llvm::StringRef name)
      : type(type), name(name){};

  /// create a JITLambda that point to the function name of the given module.
  static llvm::Expected<std::unique_ptr<JITLambda>>
  create(llvm::StringRef name, mlir::ModuleOp &module,
         llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline);

  /// invokeRaw execute the jit lambda with a list of Argument, the last one is
  /// used to store the result of the computation.
  /// Example:
  /// uin64_t arg0 = 1;
  /// uin64_t res;
  /// llvm::SmallVector<void *> args{&arg1, &res};
  /// lambda.invokeRaw(args);
  llvm::Error invokeRaw(llvm::MutableArrayRef<void *> args);

  /// invoke the jit lambda with the Argument.
  llvm::Error invoke(Argument &args);

private:
  mlir::LLVM::LLVMFunctionType type;
  llvm::StringRef name;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

} // namespace zamalang
} // namespace mlir

#endif
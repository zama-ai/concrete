#ifndef ZAMALANG_SUPPORT_COMPILERTOOLS_H_
#define ZAMALANG_SUPPORT_COMPILERTOOLS_H_

#include <mlir/Dialect/LLVMIR/LLVMTypes.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/Pass/PassManager.h>

namespace mlir {
namespace zamalang {

class CompilerTools {
public:
  /// lowerHLFHEToMlirLLVMDialect run all passes to lower FHE dialects to mlir
  /// LLVM dialect.
  static mlir::LogicalResult lowerHLFHEToMlirLLVMDialect(
      mlir::MLIRContext &context, mlir::Operation *module,
      llvm::function_ref<bool(std::string)> enablePass = [](std::string pass) {
        return true;
      });

  static llvm::Expected<std::unique_ptr<llvm::Module>>
  toLLVMModule(llvm::LLVMContext &context, mlir::ModuleOp &module,
               llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline);
};

/// JITLambda is a tool to JIT compile an mlir module and to invoke a function
/// of the module.
class JITLambda {
public:
  JITLambda(mlir::LLVM::LLVMFunctionType type, llvm::StringRef name)
      : type(type), name(name){};

  /// create a JITLambda that point to the function name of the given module.
  static llvm::Expected<std::unique_ptr<JITLambda>>
  create(llvm::StringRef name, mlir::ModuleOp &module,
         llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline);

  /// invokeRaw execute the jit lambda with a lits of arguments, the last one is
  /// used to store the result of the computation.
  /// Example:
  /// uin64_t arg0 = 1;
  /// uin64_t res;
  /// llvm::SmallVector<void *> args{&arg1, &res};
  /// lambda.invokeRaw(args);
  llvm::Error invokeRaw(llvm::MutableArrayRef<void *> args);

private:
  mlir::LLVM::LLVMFunctionType type;
  llvm::StringRef name;
  std::unique_ptr<mlir::ExecutionEngine> engine;
};

} // namespace zamalang
} // namespace mlir

#endif
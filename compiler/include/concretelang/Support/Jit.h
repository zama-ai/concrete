// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef COMPILER_JIT_H
#define COMPILER_JIT_H

#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LogicalResult.h>

#include <concretelang/ClientLib/KeySet.h>
#include <concretelang/ClientLib/PublicArguments.h>

namespace mlir {
namespace concretelang {

using ::concretelang::clientlib::CircuitGate;
using ::concretelang::clientlib::KeySet;
namespace clientlib = ::concretelang::clientlib;

/// JITLambda is a tool to JIT compile an mlir module and to invoke a function
/// of the module.
class JITLambda {
public:
  JITLambda(mlir::LLVM::LLVMFunctionType type, llvm::StringRef name)
      : type(type), name(name){};

  /// create a JITLambda that point to the function name of the given module.
  /// Use runtimeLibPath as a shared library if specified.
  static llvm::Expected<std::unique_ptr<JITLambda>>
  create(llvm::StringRef name, mlir::ModuleOp &module,
         llvm::function_ref<llvm::Error(llvm::Module *)> optPipeline,
         llvm::Optional<std::string> runtimeLibPath = {});

  /// Call the JIT lambda with the public arguments.
  llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
  call(clientlib::PublicArguments &args);

  void setUseDataflow(bool option) { this->useDataflow = option; }

private:
  /// invokeRaw execute the jit lambda with a list of Argument, the last one is
  /// used to store the result of the computation.
  /// Example:
  /// uin64_t arg0 = 1;
  /// uin64_t res;
  /// llvm::SmallVector<void *> args{&arg1, &res};
  /// lambda.invokeRaw(args);
  llvm::Error invokeRaw(llvm::MutableArrayRef<void *> args);

private:
  mlir::LLVM::LLVMFunctionType type;
  std::string name;
  std::unique_ptr<mlir::ExecutionEngine> engine;
  // Tell if the DF parallelization was on or during compilation. This will be
  // useful to abort execution if the runtime doesn't support dataflow
  // execution, instead of having undefined symbol issues
  bool useDataflow = false;
};

} // namespace concretelang
} // namespace mlir

#endif // COMPILER_JIT_H

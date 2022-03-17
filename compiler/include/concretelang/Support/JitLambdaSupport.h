// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_JITLAMBDA_SUPPORT
#define CONCRETELANG_SUPPORT_JITLAMBDA_SUPPORT

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/ExecutionEngine/ExecutionEngine.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

#include <concretelang/Support/CompilerEngine.h>
#include <concretelang/Support/Jit.h>
#include <concretelang/Support/LambdaSupport.h>

namespace mlir {
namespace concretelang {

namespace clientlib = ::concretelang::clientlib;

/// JitCompilationResult is the result of a Jit compilation, the server JIT
/// lambda and the clientParameters.
struct JitCompilationResult {
  std::unique_ptr<concretelang::JITLambda> lambda;
  clientlib::ClientParameters clientParameters;
};

/// JitLambdaSupport is the instantiated LambdaSupport for the Jit Compilation.
class JitLambdaSupport
    : public LambdaSupport<concretelang::JITLambda *, JitCompilationResult> {

public:
  JitLambdaSupport(
      llvm::Optional<llvm::StringRef> runtimeLibPath = llvm::None,
      llvm::function_ref<llvm::Error(llvm::Module *)> llvmOptPipeline =
          mlir::makeOptimizingTransformer(3, 0, nullptr));

  llvm::Expected<std::unique_ptr<JitCompilationResult>>
  compile(llvm::SourceMgr &program, CompilationOptions options) override;
  using LambdaSupport::compile;

  llvm::Expected<concretelang::JITLambda *>
  loadServerLambda(JitCompilationResult &result) override {
    return result.lambda.get();
  }

  llvm::Expected<clientlib::ClientParameters>
  loadClientParameters(JitCompilationResult &result) override {
    return result.clientParameters;
  }

  llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
  serverCall(concretelang::JITLambda *lambda,
             clientlib::PublicArguments &args) override {
    return lambda->call(args);
  }

private:
  llvm::Optional<llvm::StringRef> runtimeLibPath;
  llvm::function_ref<llvm::Error(llvm::Module *)> llvmOptPipeline;
};

} // namespace concretelang
} // namespace mlir

#endif

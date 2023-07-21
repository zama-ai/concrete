// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_JIT_SUPPORT
#define CONCRETELANG_SUPPORT_JIT_SUPPORT

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
  std::shared_ptr<concretelang::JITLambda> lambda;
  clientlib::ClientParameters clientParameters;
  CompilationFeedback feedback;
};

/// JITSupport is the instantiated LambdaSupport for the Jit Compilation.
class JITSupport
    : public LambdaSupport<std::shared_ptr<concretelang::JITLambda>,
                           JitCompilationResult> {

public:
  JITSupport(std::optional<std::string> runtimeLibPath = std::nullopt);

  llvm::Expected<std::unique_ptr<JitCompilationResult>>
  compile(llvm::SourceMgr &program, CompilationOptions options) override;
  llvm::Expected<std::unique_ptr<JitCompilationResult>>
  compile(mlir::ModuleOp program,
          std::shared_ptr<mlir::concretelang::CompilationContext> cctx,
          CompilationOptions options) override;
  using LambdaSupport::compile;

  llvm::Expected<std::shared_ptr<concretelang::JITLambda>>
  loadServerLambda(JitCompilationResult &result) override {
    return result.lambda;
  }

  llvm::Expected<clientlib::ClientParameters>
  loadClientParameters(JitCompilationResult &result) override {
    return result.clientParameters;
  }

  llvm::Expected<CompilationFeedback>
  loadCompilationFeedback(JitCompilationResult &result) override {
    return result.feedback;
  }

  llvm::Expected<std::unique_ptr<clientlib::PublicResult>>
  serverCall(std::shared_ptr<concretelang::JITLambda> lambda,
             clientlib::PublicArguments &args,
             clientlib::EvaluationKeys &evaluationKeys) override {
    return lambda->call(args, evaluationKeys);
  }

private:
  template <typename T>
  llvm::Expected<std::unique_ptr<JitCompilationResult>>
  compileWithEngine(T program, CompilationOptions options,
                    concretelang::CompilerEngine &engine);

  std::optional<std::string> runtimeLibPath;
  llvm::function_ref<llvm::Error(llvm::Module *)> llvmOptPipeline;
};

} // namespace concretelang
} // namespace mlir

#endif

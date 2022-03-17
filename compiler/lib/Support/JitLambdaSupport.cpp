// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <concretelang/Support/JitLambdaSupport.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

namespace mlir {
namespace concretelang {

JitLambdaSupport::JitLambdaSupport(
    llvm::Optional<llvm::StringRef> runtimeLibPath,
    llvm::function_ref<llvm::Error(llvm::Module *)> llvmOptPipeline)
    : runtimeLibPath(runtimeLibPath), llvmOptPipeline(llvmOptPipeline) {}

llvm::Expected<std::unique_ptr<JitCompilationResult>>
JitLambdaSupport::compile(llvm::SourceMgr &program,
                          CompilationOptions options) {

  // Setup the compiler engine
  auto context = std::make_shared<CompilationContext>();
  concretelang::CompilerEngine engine(context);

  engine.setCompilationOptions(options);

  // Compile to LLVM Dialect
  auto compilationResult =
      engine.compile(program, CompilerEngine::Target::LLVM_IR);

  if (auto err = compilationResult.takeError()) {
    return std::move(err);
  }

  if (!options.clientParametersFuncName.hasValue()) {
    return StreamStringError("Need to have a funcname to JIT compile");
  }

  // Compile from LLVM Dialect to JITLambda
  auto mlirModule = compilationResult.get().mlirModuleRef->get();
  auto lambda = concretelang::JITLambda::create(
      *options.clientParametersFuncName, mlirModule, llvmOptPipeline,
      runtimeLibPath);
  if (auto err = lambda.takeError()) {
    return std::move(err);
  }

  if (!compilationResult.get().clientParameters.hasValue()) {
    // i.e. that should not occurs
    return StreamStringError("No client parameters has been generated");
  }
  auto result = std::make_unique<JitCompilationResult>();
  result->lambda = std::move(*lambda);
  result->clientParameters =
      compilationResult.get().clientParameters.getValue();
  return std::move(result);
}

} // namespace concretelang
} // namespace mlir
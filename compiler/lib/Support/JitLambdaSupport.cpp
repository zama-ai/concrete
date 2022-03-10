// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <concretelang/Support/JitLambdaSupport.h>

namespace mlir {
namespace concretelang {

llvm::Expected<std::unique_ptr<JitCompilationResult>>
JitLambdaSupport::compile(llvm::SourceMgr &program, std::string funcname) {
  // Setup the compiler engine
  auto context = std::make_shared<CompilationContext>();
  concretelang::CompilerEngine engine(context);

  // We need client parameters to be generated
  engine.setGenerateClientParameters(true);
  engine.setClientParametersFuncName(funcname);

  // Compile to LLVM Dialect
  auto compilationResult =
      engine.compile(program, CompilerEngine::Target::LLVM_IR);

  if (auto err = compilationResult.takeError()) {
    return std::move(err);
  }

  // Compile from LLVM Dialect to JITLambda
  auto mlirModule = compilationResult.get().mlirModuleRef->get();
  auto lambda = concretelang::JITLambda::create(
      funcname, mlirModule, llvmOptPipeline, runtimeLibPath);
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
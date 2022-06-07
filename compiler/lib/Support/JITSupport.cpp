// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Support/JITSupport.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

namespace mlir {
namespace concretelang {

JITSupport::JITSupport(llvm::Optional<std::string> runtimeLibPath)
    : runtimeLibPath(runtimeLibPath) {}

llvm::Expected<std::unique_ptr<JitCompilationResult>>
JITSupport::compile(llvm::SourceMgr &program, CompilationOptions options) {
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
      *options.clientParametersFuncName, mlirModule,
      mlir::makeOptimizingTransformer(3, 0, nullptr), runtimeLibPath);
  if (auto err = lambda.takeError()) {
    return std::move(err);
  }
  if (!compilationResult.get().clientParameters.hasValue()) {
    // i.e. that should not occurs
    return StreamStringError("No client parameters has been generated");
  }
  auto result = std::make_unique<JitCompilationResult>();
  result->lambda = std::shared_ptr<concretelang::JITLambda>(std::move(*lambda));
  // Mark the lambda as compiled using DF parallelization
  result->lambda->setUseDataflow(options.dataflowParallelize ||
                                 options.autoParallelize);
  result->clientParameters =
      compilationResult.get().clientParameters.getValue();
  return std::move(result);
}

} // namespace concretelang
} // namespace mlir
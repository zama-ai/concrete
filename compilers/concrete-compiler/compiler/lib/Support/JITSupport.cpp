// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Runtime/DFRuntime.hpp>
#include <concretelang/Support/JITSupport.h>
#include <llvm/Support/TargetSelect.h>
#include <mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h>

namespace mlir {
namespace concretelang {

JITSupport::JITSupport(std::optional<std::string> runtimeLibPath)
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

  if (!options.mainFuncName.has_value()) {
    return StreamStringError("Need to have a funcname to JIT compile");
  }
  // Compile from LLVM Dialect to JITLambda
  auto mlirModule = compilationResult.get().mlirModuleRef->get();
  auto lambda = concretelang::JITLambda::create(
      *options.mainFuncName, mlirModule,
      mlir::makeOptimizingTransformer(3, 0, nullptr), runtimeLibPath);
  if (auto err = lambda.takeError()) {
    return std::move(err);
  }
  if (!compilationResult.get().programInfo.has_value()) {
    // i.e. that should not occurs
    return StreamStringError("No program info has been generated");
  }
  auto result = std::make_unique<JitCompilationResult>();
  result->lambda = std::shared_ptr<concretelang::JITLambda>(std::move(*lambda));
  // Mark the lambda as compiled using DF parallelization
  result->lambda->setUseDataflow(options.dataflowParallelize ||
                                 options.autoParallelize);
  if (!mlir::concretelang::dfr::_dfr_is_root_node()) {
    result->clientParameters = clientlib::ClientParameters();
  } else {
    result->clientParameters = ClientParameters::fromProgramInfo(
        compilationResult.get().programInfo.value());
    result->feedback = compilationResult.get().feedback.value();
  }
  return std::move(result);
}

} // namespace concretelang
} // namespace mlir

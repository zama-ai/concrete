// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concrete-protocol.pb.h"
#include <concretelang/Runtime/DFRuntime.hpp>
#include <concretelang/Support/JITSupport.h>
#include <llvm/Support/TargetSelect.h>
#include <memory>
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

  engine.setCompilationOptions(std::move(options));
  // Compile to LLVM Dialect
  auto compilationResult =
      engine.compile(program, CompilerEngine::Target::LLVM_IR);

  if (auto err = compilationResult.takeError()) {
    return std::move(err);
  }

  if (!engine.getCompilationOptions().clientParametersFuncName.has_value()) {
    return StreamStringError("Need to have a funcname to JIT compile");
  }
  // Compile from LLVM Dialect to JITLambda
  auto mlirModule = compilationResult.get().mlirModuleRef->get();
  auto lambda = concretelang::JITLambda::create(
      *engine.getCompilationOptions().clientParametersFuncName, mlirModule,
      mlir::makeOptimizingTransformer(3, 0, nullptr), runtimeLibPath);
  if (auto err = lambda.takeError()) {
    return std::move(err);
  }
  if (!compilationResult.get().clientParameters) {
    // i.e. that should not occurs
    return StreamStringError("No client parameters has been generated");
  }
  auto result = std::make_unique<JitCompilationResult>();
  result->lambda = std::shared_ptr<concretelang::JITLambda>(std::move(*lambda));
  // Mark the lambda as compiled using DF parallelization
  result->lambda->setUseDataflow(engine.getCompilationOptions().dataflowParallelize ||
                                 engine.getCompilationOptions().autoParallelize);
  if (!mlir::concretelang::dfr::_dfr_is_root_node()) {
    result->clientParameters = std::unique_ptr<protocol::ProgramInfo>();
  } else {
    result->clientParameters = std::move(compilationResult.get().clientParameters);
    result->feedback = compilationResult.get().feedback.value();
  }
  return std::move(result);
}

} // namespace concretelang
} // namespace mlir

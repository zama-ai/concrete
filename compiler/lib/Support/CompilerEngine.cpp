#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Conversion/Passes.h"
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

namespace mlir {
namespace zamalang {

void CompilerEngine::loadDialects() {
  context->getOrLoadDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  context->getOrLoadDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();
  context->getOrLoadDialect<mlir::zamalang::LowLFHE::LowLFHEDialect>();
  context->getOrLoadDialect<mlir::StandardOpsDialect>();
  context->getOrLoadDialect<mlir::memref::MemRefDialect>();
  context->getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context->getOrLoadDialect<mlir::LLVM::LLVMDialect>();
}

std::string CompilerEngine::getCompiledModule() {
  std::string compiledModule;
  llvm::raw_string_ostream os(compiledModule);
  module_ref->print(os);
  return os.str();
}

llvm::Expected<mlir::LogicalResult>
CompilerEngine::compileFHE(std::string mlir_input) {
  module_ref = mlir::parseSourceString(mlir_input, context);
  if (!module_ref) {
    return llvm::make_error<llvm::StringError>("mlir parsing failed",
                                               llvm::inconvertibleErrorCode());
  }
  mlir::zamalang::V0FHEContext fheContext;
  // Lower to MLIR Std
  if (mlir::zamalang::CompilerTools::lowerHLFHEToMlirStdsDialect(
          *context, module_ref.get(), fheContext)
          .failed()) {
    return llvm::make_error<llvm::StringError>("failed to lower to MLIR Std",
                                               llvm::inconvertibleErrorCode());
  }
  // Create the client parameters
  auto clientParameter = mlir::zamalang::createClientParametersForV0(
      fheContext, "main", module_ref.get());
  if (auto err = clientParameter.takeError()) {
    return llvm::make_error<llvm::StringError>(
        "cannot generate client parameters", llvm::inconvertibleErrorCode());
  }
  auto maybeKeySet =
      mlir::zamalang::KeySet::generate(clientParameter.get(), 0, 0);
  if (auto err = maybeKeySet.takeError()) {
    return llvm::make_error<llvm::StringError>("cannot generate keyset",
                                               llvm::inconvertibleErrorCode());
  }
  keySet = std::move(maybeKeySet.get());

  // Lower to MLIR LLVM Dialect
  if (mlir::zamalang::CompilerTools::lowerMlirStdsDialectToMlirLLVMDialect(
          *context, module_ref.get())
          .failed()) {
    return llvm::make_error<llvm::StringError>(
        "failed to lower to LLVM dialect", llvm::inconvertibleErrorCode());
  }
  return mlir::success();
}

llvm::Expected<uint64_t> CompilerEngine::run(std::vector<uint64_t> args) {
  // Create the JIT lambda
  auto defaultOptPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  auto module = module_ref.get();
  auto maybeLambda =
      mlir::zamalang::JITLambda::create("main", module, defaultOptPipeline);
  if (!maybeLambda) {
    return llvm::make_error<llvm::StringError>("couldn't create lambda",
                                               llvm::inconvertibleErrorCode());
  }
  auto lambda = std::move(maybeLambda.get());

  // Create the arguments of the JIT lambda
  auto maybeArguments = mlir::zamalang::JITLambda::Argument::create(*keySet);
  if (auto err = maybeArguments.takeError()) {
    return llvm::make_error<llvm::StringError>("cannot create lambda args",
                                               llvm::inconvertibleErrorCode());
  }
  // Set the arguments
  auto arguments = std::move(maybeArguments.get());
  for (auto i = 0; i < args.size(); i++) {
    if (auto err = arguments->setArg(i, args[i])) {
      return llvm::make_error<llvm::StringError>(
          "cannot push argument", llvm::inconvertibleErrorCode());
    }
  }
  // Invoke the lambda
  if (lambda->invoke(*arguments)) {
    return llvm::make_error<llvm::StringError>("failed execution",
                                               llvm::inconvertibleErrorCode());
  }
  uint64_t res = 0;
  if (auto err = arguments->getResult(0, res)) {
    return llvm::make_error<llvm::StringError>("cannot get result",
                                               llvm::inconvertibleErrorCode());
  }
  return res;
}
} // namespace zamalang
} // namespace mlir
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/Dialect/Linalg/IR/LinalgOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/StandardOps/IR/Ops.h>
#include <mlir/ExecutionEngine/OptUtils.h>
#include <mlir/Parser.h>

#include <zamalang/Dialect/HLFHE/IR/HLFHEDialect.h>
#include <zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h>
#include <zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h>
#include <zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h>
#include <zamalang/Support/CompilerEngine.h>
#include <zamalang/Support/Pipeline.h>

namespace mlir {
namespace zamalang {

void CompilerEngine::loadDialects() {
  context->getOrLoadDialect<mlir::zamalang::HLFHELinalg::HLFHELinalgDialect>();
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

llvm::Error CompilerEngine::compile(
    std::string mlirStr,
    llvm::Optional<mlir::zamalang::V0FHEConstraint> overrideConstraints) {
  module_ref = mlir::parseSourceString(mlirStr, context);
  if (!module_ref) {
    return llvm::make_error<llvm::StringError>("mlir parsing failed",
                                               llvm::inconvertibleErrorCode());
  }

  mlir::ModuleOp module = module_ref.get();

  llvm::Optional<mlir::zamalang::V0FHEConstraint> fheConstraintsOpt =
      overrideConstraints;

  if (!fheConstraintsOpt.hasValue()) {
    llvm::Expected<llvm::Optional<mlir::zamalang::V0FHEConstraint>>
        fheConstraintsOrErr =
            mlir::zamalang::pipeline::getFHEConstraintsFromHLFHE(*context,
                                                                 module);

    if (auto err = fheConstraintsOrErr.takeError())
      return std::move(err);

    if (!fheConstraintsOrErr.get().hasValue()) {
      return llvm::make_error<llvm::StringError>(
          "Could not determine maximum required precision for encrypted "
          "integers "
          "and maximum value for the Minimal Arithmetic Noise Padding",
          llvm::inconvertibleErrorCode());
    }

    fheConstraintsOpt = fheConstraintsOrErr.get();
  }

  mlir::zamalang::V0FHEConstraint fheConstraints = fheConstraintsOpt.getValue();
  const mlir::zamalang::V0Parameter *parameter = getV0Parameter(fheConstraints);

  if (!parameter) {
    std::string buffer;
    llvm::raw_string_ostream strs(buffer);
    strs << "Could not determine V0 parameters for 2-norm of "
         << fheConstraints.norm2 << " and p of " << fheConstraints.p;

    return llvm::make_error<llvm::StringError>(strs.str(),
                                               llvm::inconvertibleErrorCode());
  }

  mlir::zamalang::V0FHEContext fheContext{fheConstraints, *parameter};

  // Lower to MLIR Std
  if (mlir::zamalang::pipeline::lowerHLFHEToStd(*context, module, fheContext,
                                                false)
          .failed()) {
    return llvm::make_error<llvm::StringError>("failed to lower to MLIR Std",
                                               llvm::inconvertibleErrorCode());
  }
  // Create the client parameters
  auto clientParameter = mlir::zamalang::createClientParametersForV0(
      fheContext, "main", module_ref.get());
  if (auto err = clientParameter.takeError()) {
    return std::move(err);
  }
  auto maybeKeySet =
      mlir::zamalang::KeySet::generate(clientParameter.get(), 0, 0);
  if (auto err = maybeKeySet.takeError()) {
    return std::move(err);
  }
  keySet = std::move(maybeKeySet.get());

  // Lower to MLIR LLVM Dialect
  if (mlir::zamalang::pipeline::lowerStdToLLVMDialect(*context, module, false)
          .failed()) {
    return llvm::make_error<llvm::StringError>(
        "failed to lower to LLVM dialect", llvm::inconvertibleErrorCode());
  }
  return llvm::Error::success();
}

llvm::Expected<std::unique_ptr<JITLambda::Argument>>
CompilerEngine::buildArgument() {
  if (keySet.get() == nullptr) {
    return llvm::make_error<llvm::StringError>(
        "CompilerEngine::buildArgument: invalid engine state, the keySet has "
        "not been generated",
        llvm::inconvertibleErrorCode());
  }
  return JITLambda::Argument::create(*keySet);
}

llvm::Error CompilerEngine::invoke(JITLambda::Argument &arg) {
  // Create the JIT lambda
  auto defaultOptPipeline = mlir::makeOptimizingTransformer(3, 0, nullptr);
  auto module = module_ref.get();
  auto maybeLambda =
      mlir::zamalang::JITLambda::create("main", module, defaultOptPipeline);
  if (auto err = maybeLambda.takeError()) {
    return std::move(err);
  }
  // Invoke the lambda
  if (auto err = maybeLambda.get()->invoke(arg)) {
    return std::move(err);
  }
  return llvm::Error::success();
}

llvm::Expected<uint64_t> CompilerEngine::run(std::vector<uint64_t> args) {
  // Build the argument of the JIT lambda.
  auto maybeArgument = buildArgument();
  if (auto err = maybeArgument.takeError()) {
    return std::move(err);
  }
  // Set the integer arguments
  auto arguments = std::move(maybeArgument.get());
  for (auto i = 0; i < args.size(); i++) {
    if (auto err = arguments->setArg(i, args[i])) {
      return std::move(err);
    }
  }
  // Invoke the lambda
  if (auto err = invoke(*arguments)) {
    return std::move(err);
  }
  uint64_t res = 0;
  if (auto err = arguments->getResult(0, res)) {
    return std::move(err);
  }
  return res;
}
} // namespace zamalang
} // namespace mlir

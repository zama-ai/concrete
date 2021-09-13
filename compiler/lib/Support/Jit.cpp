#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>

#include <zamalang/Support/Jit.h>
#include <zamalang/Support/logging.h>

namespace mlir {
namespace zamalang {

// JIT-compiles `module` invokes `func` with the arguments passed in
// `jitArguments` and `keySet`
mlir::LogicalResult
runJit(mlir::ModuleOp module, llvm::StringRef func,
       llvm::ArrayRef<uint64_t> funcArgs, mlir::zamalang::KeySet &keySet,
       std::function<llvm::Error(llvm::Module *)> optPipeline,
       llvm::raw_ostream &os) {
  // Create the JIT lambda
  auto maybeLambda =
      mlir::zamalang::JITLambda::create(func, module, optPipeline);
  if (!maybeLambda) {
    return mlir::failure();
  }
  auto lambda = std::move(maybeLambda.get());

  // Create the arguments of the JIT lambda
  auto maybeArguments = mlir::zamalang::JITLambda::Argument::create(keySet);
  if (auto err = maybeArguments.takeError()) {
    ::mlir::zamalang::log_error()
        << "Cannot create lambda arguments: " << err << "\n";
    return mlir::failure();
  }

  // Set the arguments
  auto arguments = std::move(maybeArguments.get());
  for (size_t i = 0; i < funcArgs.size(); i++) {
    if (auto err = arguments->setArg(i, funcArgs[i])) {
      ::mlir::zamalang::log_error()
          << "Cannot push argument " << i << ": " << err << "\n";
      return mlir::failure();
    }
  }
  // Invoke the lambda
  if (auto err = lambda->invoke(*arguments)) {
    ::mlir::zamalang::log_error() << "Cannot invoke : " << err << "\n";
    return mlir::failure();
  }
  uint64_t res = 0;
  if (auto err = arguments->getResult(0, res)) {
    ::mlir::zamalang::log_error() << "Cannot get result : " << err << "\n";
    return mlir::failure();
  }
  llvm::errs() << res << "\n";
  return mlir::success();
}

} // namespace zamalang
} // namespace mlir

#include "zamalang-c/Support/CompilerEngine.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/ExecutionArgument.h"

using mlir::zamalang::CompilerEngine;
using mlir::zamalang::ExecutionArgument;

void compilerEngineCompile(compilerEngine engine, const char *module) {
  auto error = engine.ptr->compile(module);
  if (error) {
    llvm::errs() << "Compilation failed: " << error << "\n";
    llvm::consumeError(std::move(error));
    throw std::runtime_error(
        "failed compiling, see previous logs for more info");
  }
}

uint64_t compilerEngineRun(compilerEngine engine, exectuionArguments args) {
  auto args_size = args.size;
  auto maybeArgument = engine.ptr->buildArgument();
  if (auto err = maybeArgument.takeError()) {
    llvm::errs() << "Execution failed: " << err << "\n";
    llvm::consumeError(std::move(err));
    throw std::runtime_error(
        "failed building arguments, see previous logs for more info");
  }
  // Set the integer/tensor arguments
  auto arguments = std::move(maybeArgument.get());
  for (auto i = 0; i < args_size; i++) {
    if (args.data[i].isInt()) { // integer argument
      if (auto err = arguments->setArg(i, args.data[i].getIntegerArgument())) {
        llvm::errs() << "Execution failed: " << err << "\n";
        llvm::consumeError(std::move(err));
        throw std::runtime_error("failed pushing integer argument, see "
                                 "previous logs for more info");
      }
    } else { // tensor argument
      assert(args.data[i].isTensor() && "should be tensor argument");
      if (auto err = arguments->setArg(i, args.data[i].getTensorArgument(),
                                       args.data[i].getTensorSize())) {
        llvm::errs() << "Execution failed: " << err << "\n";
        llvm::consumeError(std::move(err));
        throw std::runtime_error("failed pushing tensor argument, see "
                                 "previous logs for more info");
      }
    }
  }
  // Invoke the lambda
  if (auto err = engine.ptr->invoke(*arguments)) {
    llvm::errs() << "Execution failed: " << err << "\n";
    llvm::consumeError(std::move(err));
    throw std::runtime_error("failed running, see previous logs for more info");
  }
  uint64_t result = 0;
  if (auto err = arguments->getResult(0, result)) {
    llvm::errs() << "Execution failed: " << err << "\n";
    llvm::consumeError(std::move(err));
    throw std::runtime_error(
        "failed getting result, see previous logs for more info");
  }
  return result;
}
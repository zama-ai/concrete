#include "zamalang-c/Support/CompilerEngine.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/ExecutionArgument.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/JitCompilerEngine.h"
#include "zamalang/Support/logging.h"

// using mlir::zamalang::CompilerEngine;
using mlir::zamalang::ExecutionArgument;
using mlir::zamalang::JitCompilerEngine;

mlir::zamalang::JitCompilerEngine::Lambda buildLambda(const char *module,
                                                      const char *funcName) {
  mlir::zamalang::JitCompilerEngine engine;
  llvm::Expected<mlir::zamalang::JitCompilerEngine::Lambda> lambdaOrErr =
      engine.buildLambda(module, funcName);
  if (!lambdaOrErr) {
    mlir::zamalang::log_error()
        << "Compilation failed: "
        << llvm::toString(std::move(lambdaOrErr.takeError())) << "\n";
    throw std::runtime_error(
        "failed compiling, see previous logs for more info");
  }
  return std::move(*lambdaOrErr);
}

uint64_t invokeLambda(lambda l, executionArguments args) {
  mlir::zamalang::JitCompilerEngine::Lambda *lambda_ptr =
      (mlir::zamalang::JitCompilerEngine::Lambda *)l.ptr;

  if (args.size != lambda_ptr->getNumArguments()) {
    throw std::invalid_argument("wrong number of arguments");
  }
  // Set the integer/tensor arguments
  std::vector<mlir::zamalang::LambdaArgument *> lambdaArgumentsRef;
  for (auto i = 0; i < args.size; i++) {
    if (args.data[i].isInt()) { // integer argument
      lambdaArgumentsRef.push_back(new mlir::zamalang::IntLambdaArgument<>(
          args.data[i].getIntegerArgument()));
    } else { // tensor argument
      llvm::MutableArrayRef<uint8_t> tensor(args.data[i].getTensorArgument(),
                                            args.data[i].getTensorSize());
      lambdaArgumentsRef.push_back(
          new mlir::zamalang::TensorLambdaArgument<
              mlir::zamalang::IntLambdaArgument<uint8_t>>(tensor));
    }
  }
  // Run lambda
  llvm::Expected<uint64_t> resOrError = (*lambda_ptr)(
      llvm::ArrayRef<mlir::zamalang::LambdaArgument *>(lambdaArgumentsRef));
  // Free heap
  for (size_t i = 0; i < lambdaArgumentsRef.size(); i++)
    delete lambdaArgumentsRef[i];

  if (!resOrError) {
    mlir::zamalang::log_error()
        << "Lambda invokation failed: "
        << llvm::toString(std::move(resOrError.takeError())) << "\n";
    throw std::runtime_error(
        "failed invoking lambda, see previous logs for more info");
  }
  return *resOrError;
}

std::string roundTrip(const char *module) {
  std::shared_ptr<mlir::zamalang::CompilationContext> ccx =
      mlir::zamalang::CompilationContext::createShared();
  mlir::zamalang::JitCompilerEngine ce{ccx};

  llvm::Expected<mlir::zamalang::CompilerEngine::CompilationResult> retOrErr =
      ce.compile(module, mlir::zamalang::CompilerEngine::Target::ROUND_TRIP);
  if (!retOrErr) {
    mlir::zamalang::log_error()
        << llvm::toString(std::move(retOrErr.takeError())) << "\n";
    throw std::runtime_error(
        "mlir parsing failed, see previous logs for more info");
  }

  std::string result;
  llvm::raw_string_ostream os(result);
  retOrErr->mlirModuleRef->get().print(os);
  return os.str();
}

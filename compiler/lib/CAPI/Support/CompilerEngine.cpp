#include "zamalang-c/Support/CompilerEngine.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/ExecutionArgument.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/JitCompilerEngine.h"

// using mlir::zamalang::CompilerEngine;
using mlir::zamalang::ExecutionArgument;
using mlir::zamalang::JitCompilerEngine;

mlir::zamalang::JitCompilerEngine::Lambda buildLambda(const char *module,
                                                      const char *funcName) {
  mlir::zamalang::JitCompilerEngine engine;
  llvm::Expected<mlir::zamalang::JitCompilerEngine::Lambda> lambdaOrErr =
      engine.buildLambda(module, funcName);
  if (!lambdaOrErr) {
    std::string backingString;
    llvm::raw_string_ostream os(backingString);
    os << "Compilation failed: "
       << llvm::toString(std::move(lambdaOrErr.takeError()));
    throw std::runtime_error(os.str());
  }
  return std::move(*lambdaOrErr);
}

lambdaArgument invokeLambda(lambda l, executionArguments args) {
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
  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> resOrError =
      (*lambda_ptr)
          .
          operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
              llvm::ArrayRef<mlir::zamalang::LambdaArgument *>(
                  lambdaArgumentsRef));
  // Free heap
  for (size_t i = 0; i < lambdaArgumentsRef.size(); i++)
    delete lambdaArgumentsRef[i];

  if (!resOrError) {
    std::string backingString;
    llvm::raw_string_ostream os(backingString);
    os << "Lambda invocation failed: "
       << llvm::toString(std::move(resOrError.takeError()));
    throw std::runtime_error(os.str());
  }
  lambdaArgument result{std::move(*resOrError)};
  return result;
}

std::string roundTrip(const char *module) {
  std::shared_ptr<mlir::zamalang::CompilationContext> ccx =
      mlir::zamalang::CompilationContext::createShared();
  mlir::zamalang::JitCompilerEngine ce{ccx};

  std::string backingString;
  llvm::raw_string_ostream os(backingString);

  llvm::Expected<mlir::zamalang::CompilerEngine::CompilationResult> retOrErr =
      ce.compile(module, mlir::zamalang::CompilerEngine::Target::ROUND_TRIP);
  if (!retOrErr) {
    os << "MLIR parsing failed: "
       << llvm::toString(std::move(retOrErr.takeError()));
    throw std::runtime_error(os.str());
  }

  retOrErr->mlirModuleRef->get().print(os);
  return os.str();
}

bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->isa<mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint64_t>>>();
}

std::vector<uint64_t> lambdaArgumentGetTensorData(lambdaArgument &lambda_arg) {
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint64_t>> *arg =
      lambda_arg.ptr->dyn_cast<mlir::zamalang::TensorLambdaArgument<
          mlir::zamalang::IntLambdaArgument<uint64_t>>>();
  if (arg == nullptr) {
    throw std::invalid_argument(
        "LambdaArgument isn't a tensor, should "
        "be a TensorLambdaArgument<IntLambdaArgument<uint64_t>>");
  }

  llvm::Expected<size_t> sizeOrErr = arg->getNumElements();
  if (!sizeOrErr) {
    std::string backingString;
    llvm::raw_string_ostream os(backingString);
    os << "Couldn't get size of tensor: "
       << llvm::toString(std::move(sizeOrErr.takeError()));
    throw std::runtime_error(os.str());
  }
  std::vector<uint64_t> data(arg->getValue(), arg->getValue() + *sizeOrErr);
  return data;
}

std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg) {
  mlir::zamalang::TensorLambdaArgument<
      mlir::zamalang::IntLambdaArgument<uint64_t>> *arg =
      lambda_arg.ptr->dyn_cast<mlir::zamalang::TensorLambdaArgument<
          mlir::zamalang::IntLambdaArgument<uint64_t>>>();
  if (arg == nullptr) {
    throw std::invalid_argument(
        "LambdaArgument isn't a tensor, should "
        "be a TensorLambdaArgument<IntLambdaArgument<uint64_t>>");
  }
  return arg->getDimensions();
}

bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->isa<mlir::zamalang::IntLambdaArgument<uint64_t>>();
}

uint64_t lambdaArgumentGetScalar(lambdaArgument &lambda_arg) {
  mlir::zamalang::IntLambdaArgument<uint64_t> *arg =
      lambda_arg.ptr->dyn_cast<mlir::zamalang::IntLambdaArgument<uint64_t>>();
  if (arg == nullptr) {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<uint64_t>");
  }
  return arg->getValue();
}

#include "llvm/ADT/SmallString.h"

#include "zamalang-c/Support/CompilerEngine.h"
#include "zamalang/Support/CompilerEngine.h"
#include "zamalang/Support/Jit.h"
#include "zamalang/Support/JitCompilerEngine.h"
#include "zamalang/Support/KeySetCache.h"

using mlir::zamalang::JitCompilerEngine;

mlir::zamalang::JitCompilerEngine::Lambda
buildLambda(const char *module, const char *funcName,
            const char *runtimeLibPath, const char *keySetCachePath) {
  // Set the runtime library path if not nullptr
  llvm::Optional<llvm::StringRef> runtimeLibPathOptional = {};
  if (runtimeLibPath != nullptr)
    runtimeLibPathOptional = runtimeLibPath;
  mlir::zamalang::JitCompilerEngine engine;

  using KeySetCache = mlir::zamalang::KeySetCache;
  using optKeySetCache = llvm::Optional<mlir::zamalang::KeySetCache>;
  auto cacheOpt = optKeySetCache();
  if (keySetCachePath != nullptr) {
    cacheOpt = KeySetCache(std::string(keySetCachePath));
  }

  llvm::Expected<mlir::zamalang::JitCompilerEngine::Lambda> lambdaOrErr =
      engine.buildLambda(module, funcName, cacheOpt, runtimeLibPathOptional);
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
    lambdaArgumentsRef.push_back(args.data[i].ptr.get());
  }
  // Run lambda
  llvm::Expected<std::unique_ptr<mlir::zamalang::LambdaArgument>> resOrError =
      (*lambda_ptr)
          .
          operator()<std::unique_ptr<mlir::zamalang::LambdaArgument>>(
              llvm::ArrayRef<mlir::zamalang::LambdaArgument *>(
                  lambdaArgumentsRef));

  if (!resOrError) {
    std::string backingString;
    llvm::raw_string_ostream os(backingString);
    os << "Lambda invocation failed: "
       << llvm::toString(std::move(resOrError.takeError()));
    throw std::runtime_error(os.str());
  }
  lambdaArgument result{std::move(*resOrError)};
  return std::move(result);
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

lambdaArgument lambdaArgumentFromTensor(std::vector<uint8_t> data,
                                        std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::zamalang::TensorLambdaArgument<
          mlir::zamalang::IntLambdaArgument<uint8_t>>>(data, dimensions)};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromScalar(uint64_t scalar) {
  lambdaArgument scalar_arg{
      std::make_shared<mlir::zamalang::IntLambdaArgument<uint64_t>>(scalar)};
  return scalar_arg;
}

template <class T>
std::runtime_error library_error(std::string prefix, llvm::Expected<T> &error) {
  return std::runtime_error(prefix + llvm::toString(error.takeError()));
}

std::string library(std::string libraryPath,
                    std::vector<std::string> mlir_modules) {
  using namespace mlir::zamalang;

  JitCompilerEngine ce{CompilationContext::createShared()};
  auto lib = ce.compile<std::string>(mlir_modules, libraryPath);
  if (!lib) {
    throw std::runtime_error("Can't link: " + llvm::toString(lib.takeError()));
  }
  return lib->sharedLibraryPath;
}

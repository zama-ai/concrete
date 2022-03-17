// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "llvm/ADT/SmallString.h"

#include "concretelang-c/Support/CompilerEngine.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/Runtime/runtime_api.h"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/Jit.h"
#include "concretelang/Support/JitCompilerEngine.h"
#include "concretelang/Support/JitLambdaSupport.h"

using mlir::concretelang::JitCompilerEngine;

#define GET_OR_THROW_LLVM_EXPECTED(VARNAME, EXPECTED)                          \
  auto VARNAME = EXPECTED;                                                     \
  if (auto err = VARNAME.takeError()) {                                        \
    throw std::runtime_error(llvm::toString(std::move(err)));                  \
  }

// JIT Support bindings ///////////////////////////////////////////////////////

MLIR_CAPI_EXPORTED JITLambdaSupport_C
jit_lambda_support(const char *runtimeLibPath) {
  llvm::StringRef str(runtimeLibPath);
  auto opt = str.empty() ? llvm::None : llvm::Optional<llvm::StringRef>(str);
  return JITLambdaSupport_C{mlir::concretelang::JitLambdaSupport(opt)};
}

std::unique_ptr<mlir::concretelang::JitCompilationResult>
jit_compile(JITLambdaSupport_C support, const char *module,
            mlir::concretelang::CompilationOptions options) {
  mlir::concretelang::JitLambdaSupport esupport;
  GET_OR_THROW_LLVM_EXPECTED(compilationResult,
                             esupport.compile(module, options));
  return std::move(*compilationResult);
}

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
jit_load_client_parameters(JITLambdaSupport_C support,
                           mlir::concretelang::JitCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(clientParameters,
                             support.support.loadClientParameters(result));
  return *clientParameters;
}

MLIR_CAPI_EXPORTED mlir::concretelang::JITLambda *
jit_load_server_lambda(JITLambdaSupport_C support,
                       mlir::concretelang::JitCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(serverLambda,
                             support.support.loadServerLambda(result));
  return *serverLambda;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
jit_server_call(JITLambdaSupport_C support,
                mlir::concretelang::JITLambda *lambda,
                concretelang::clientlib::PublicArguments &args) {
  GET_OR_THROW_LLVM_EXPECTED(publicResult,
                             support.support.serverCall(lambda, args));
  return std::move(*publicResult);
}

// Library Support bindings ///////////////////////////////////////////////////
MLIR_CAPI_EXPORTED LibraryLambdaSupport_C
library_lambda_support(const char *outputPath) {
  return LibraryLambdaSupport_C{
      mlir::concretelang::LibraryLambdaSupport(outputPath)};
}

std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibraryLambdaSupport_C support, const char *module,
                mlir::concretelang::CompilationOptions options) {
  GET_OR_THROW_LLVM_EXPECTED(compilationResult,
                             support.support.compile(module, options));
  return std::move(*compilationResult);
}

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
library_load_client_parameters(
    LibraryLambdaSupport_C support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(clientParameters,
                             support.support.loadClientParameters(result));
  return *clientParameters;
}

MLIR_CAPI_EXPORTED concretelang::serverlib::ServerLambda
library_load_server_lambda(
    LibraryLambdaSupport_C support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(serverLambda,
                             support.support.loadServerLambda(result));
  return *serverLambda;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_server_call(LibraryLambdaSupport_C support,
                    concretelang::serverlib::ServerLambda lambda,
                    concretelang::clientlib::PublicArguments &args) {
  GET_OR_THROW_LLVM_EXPECTED(publicResult,
                             support.support.serverCall(lambda, args));
  return std::move(*publicResult);
}

// Client Support bindings ///////////////////////////////////////////////////

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
key_set(concretelang::clientlib::ClientParameters clientParameters,
        llvm::Optional<concretelang::clientlib::KeySetCache> cache) {
  GET_OR_THROW_LLVM_EXPECTED(
      ks, (mlir::concretelang::LambdaSupport<int, int>::keySet(clientParameters,
                                                               cache)));
  return std::move(*ks);
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
encrypt_arguments(concretelang::clientlib::ClientParameters clientParameters,
                  concretelang::clientlib::KeySet &keySet,
                  llvm::ArrayRef<mlir::concretelang::LambdaArgument *> args) {
  GET_OR_THROW_LLVM_EXPECTED(
      publicArguments,
      (mlir::concretelang::LambdaSupport<int, int>::exportArguments(
          clientParameters, keySet, args)));
  return std::move(*publicArguments);
}

MLIR_CAPI_EXPORTED lambdaArgument
decrypt_result(concretelang::clientlib::KeySet &keySet,
               concretelang::clientlib::PublicResult &publicResult) {
  GET_OR_THROW_LLVM_EXPECTED(
      result, mlir::concretelang::typedResult<
                  std::unique_ptr<mlir::concretelang::LambdaArgument>>(
                  keySet, publicResult));
  lambdaArgument result_{std::move(*result)};
  return std::move(result_);
}

mlir::concretelang::JitCompilerEngine::Lambda
buildLambda(const char *module, const char *funcName,
            const char *runtimeLibPath, const char *keySetCachePath,
            bool autoParallelize, bool loopParallelize, bool dfParallelize) {
  // Set the runtime library path if not nullptr
  llvm::Optional<llvm::StringRef> runtimeLibPathOptional = {};
  if (runtimeLibPath != nullptr)
    runtimeLibPathOptional = runtimeLibPath;
  mlir::concretelang::JitCompilerEngine engine;

  // Set parallelization flags
  engine.setAutoParallelize(autoParallelize);
  engine.setLoopParallelize(loopParallelize);
  engine.setDataflowParallelize(dfParallelize);

  using KeySetCache = mlir::concretelang::KeySetCache;
  using optKeySetCache = llvm::Optional<mlir::concretelang::KeySetCache>;
  auto cacheOpt = optKeySetCache();
  if (keySetCachePath != nullptr) {
    cacheOpt = KeySetCache(std::string(keySetCachePath));
  }

  llvm::Expected<mlir::concretelang::JitCompilerEngine::Lambda> lambdaOrErr =
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

void terminateParallelization() {
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
  _dfr_terminate();
#else
  throw std::runtime_error(
      "This package was built without parallelization support");
#endif
}

lambdaArgument invokeLambda(lambda l, executionArguments args) {
  mlir::concretelang::JitCompilerEngine::Lambda *lambda_ptr =
      (mlir::concretelang::JitCompilerEngine::Lambda *)l.ptr;

  if (args.size != lambda_ptr->getNumArguments()) {
    throw std::invalid_argument("wrong number of arguments");
  }
  // Set the integer/tensor arguments
  std::vector<mlir::concretelang::LambdaArgument *> lambdaArgumentsRef;
  for (auto i = 0u; i < args.size; i++) {
    lambdaArgumentsRef.push_back(args.data[i].ptr.get());
  }
  // Run lambda
  llvm::Expected<std::unique_ptr<mlir::concretelang::LambdaArgument>>
      resOrError =
          (*lambda_ptr)
              .
              operator()<std::unique_ptr<mlir::concretelang::LambdaArgument>>(
                  llvm::ArrayRef<mlir::concretelang::LambdaArgument *>(
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
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::JitCompilerEngine ce{ccx};

  std::string backingString;
  llvm::raw_string_ostream os(backingString);

  llvm::Expected<mlir::concretelang::CompilerEngine::CompilationResult>
      retOrErr = ce.compile(
          module, mlir::concretelang::CompilerEngine::Target::ROUND_TRIP);
  if (!retOrErr) {
    os << "MLIR parsing failed: "
       << llvm::toString(std::move(retOrErr.takeError()));
    throw std::runtime_error(os.str());
  }

  retOrErr->mlirModuleRef->get().print(os);
  return os.str();
}

bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>>>();
}

std::vector<uint64_t> lambdaArgumentGetTensorData(lambdaArgument &lambda_arg) {
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>> *arg =
      lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint64_t>>>();
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
  mlir::concretelang::TensorLambdaArgument<
      mlir::concretelang::IntLambdaArgument<uint64_t>> *arg =
      lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint64_t>>>();
  if (arg == nullptr) {
    throw std::invalid_argument(
        "LambdaArgument isn't a tensor, should "
        "be a TensorLambdaArgument<IntLambdaArgument<uint64_t>>");
  }
  return arg->getDimensions();
}

bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->isa<mlir::concretelang::IntLambdaArgument<uint64_t>>();
}

uint64_t lambdaArgumentGetScalar(lambdaArgument &lambda_arg) {
  mlir::concretelang::IntLambdaArgument<uint64_t> *arg =
      lambda_arg.ptr
          ->dyn_cast<mlir::concretelang::IntLambdaArgument<uint64_t>>();
  if (arg == nullptr) {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<uint64_t>");
  }
  return arg->getValue();
}

lambdaArgument lambdaArgumentFromTensorU8(std::vector<uint8_t> data,
                                          std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint8_t>>>(data, dimensions)};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorU16(std::vector<uint16_t> data,
                                           std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint16_t>>>(data, dimensions)};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorU32(std::vector<uint32_t> data,
                                           std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint32_t>>>(data, dimensions)};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorU64(std::vector<uint64_t> data,
                                           std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint64_t>>>(data, dimensions)};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromScalar(uint64_t scalar) {
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::IntLambdaArgument<uint64_t>>(
          scalar)};
  return scalar_arg;
}

template <class T>
std::runtime_error library_error(std::string prefix, llvm::Expected<T> &error) {
  return std::runtime_error(prefix + llvm::toString(error.takeError()));
}

std::string library(std::string libraryPath,
                    std::vector<std::string> mlir_modules) {
  using namespace mlir::concretelang;

  JitCompilerEngine ce{CompilationContext::createShared()};
  auto lib = ce.compile(mlir_modules, libraryPath);
  if (!lib) {
    throw std::runtime_error("Can't link: " + llvm::toString(lib.takeError()));
  }
  return lib->sharedLibraryPath;
}

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
#include "concretelang/Support/JitLambdaSupport.h"

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

void terminateParallelization() {
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
  _dfr_terminate();
#else
  throw std::runtime_error(
      "This package was built without parallelization support");
#endif
}

std::string roundTrip(const char *module) {
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};

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

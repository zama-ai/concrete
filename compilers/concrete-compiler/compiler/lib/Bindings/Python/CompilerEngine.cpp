// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/ADT/SmallString.h"

#include "concretelang/Bindings/Python/CompilerEngine.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Support/CompilerEngine.h"
#include "concretelang/Support/JITSupport.h"
#include "concretelang/Support/Jit.h"

#define GET_OR_THROW_LLVM_EXPECTED(VARNAME, EXPECTED)                          \
  auto VARNAME = EXPECTED;                                                     \
  if (auto err = VARNAME.takeError()) {                                        \
    throw std::runtime_error(llvm::toString(std::move(err)));                  \
  }

// JIT Support bindings ///////////////////////////////////////////////////////

MLIR_CAPI_EXPORTED JITSupport_Py jit_support(std::string runtimeLibPath) {
  auto opt = runtimeLibPath.empty()
                 ? std::nullopt
                 : std::optional<std::string>(runtimeLibPath);
  return JITSupport_Py{mlir::concretelang::JITSupport(opt)};
}

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::JitCompilationResult>
jit_compile(JITSupport_Py support, const char *module,
            mlir::concretelang::CompilationOptions options) {
  GET_OR_THROW_LLVM_EXPECTED(compilationResult,
                             support.support.compile(module, options));
  return std::move(*compilationResult);
}

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
jit_load_client_parameters(JITSupport_Py support,
                           mlir::concretelang::JitCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(clientParameters,
                             support.support.loadClientParameters(result));
  return *clientParameters;
}

MLIR_CAPI_EXPORTED mlir::concretelang::CompilationFeedback
jit_load_compilation_feedback(
    JITSupport_Py support, mlir::concretelang::JitCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(compilationFeedback,
                             support.support.loadCompilationFeedback(result));
  return *compilationFeedback;
}

MLIR_CAPI_EXPORTED std::shared_ptr<mlir::concretelang::JITLambda>
jit_load_server_lambda(JITSupport_Py support,
                       mlir::concretelang::JitCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(serverLambda,
                             support.support.loadServerLambda(result));
  return *serverLambda;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
jit_server_call(JITSupport_Py support, mlir::concretelang::JITLambda &lambda,
                concretelang::clientlib::PublicArguments &args,
                concretelang::clientlib::EvaluationKeys &evaluationKeys) {
  GET_OR_THROW_LLVM_EXPECTED(publicResult, lambda.call(args, evaluationKeys));
  return std::move(*publicResult);
}

// Library Support bindings ///////////////////////////////////////////////////
MLIR_CAPI_EXPORTED LibrarySupport_Py
library_support(const char *outputPath, const char *runtimeLibraryPath,
                bool generateSharedLib, bool generateStaticLib,
                bool generateClientParameters, bool generateCompilationFeedback,
                bool generateCppHeader) {
  return LibrarySupport_Py{mlir::concretelang::LibrarySupport(
      outputPath, runtimeLibraryPath, generateSharedLib, generateStaticLib,
      generateClientParameters, generateCompilationFeedback,
      generateCppHeader)};
}

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibrarySupport_Py support, const char *module,
                mlir::concretelang::CompilationOptions options) {
  GET_OR_THROW_LLVM_EXPECTED(compilationResult,
                             support.support.compile(module, options));
  return std::move(*compilationResult);
}

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
library_load_client_parameters(
    LibrarySupport_Py support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(clientParameters,
                             support.support.loadClientParameters(result));
  return *clientParameters;
}

MLIR_CAPI_EXPORTED mlir::concretelang::CompilationFeedback
library_load_compilation_feedback(
    LibrarySupport_Py support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(compilationFeedback,
                             support.support.loadCompilationFeedback(result));
  return *compilationFeedback;
}

MLIR_CAPI_EXPORTED concretelang::serverlib::ServerLambda
library_load_server_lambda(
    LibrarySupport_Py support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_LLVM_EXPECTED(serverLambda,
                             support.support.loadServerLambda(result));
  return *serverLambda;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_server_call(LibrarySupport_Py support,
                    concretelang::serverlib::ServerLambda lambda,
                    concretelang::clientlib::PublicArguments &args,
                    concretelang::clientlib::EvaluationKeys &evaluationKeys) {
  GET_OR_THROW_LLVM_EXPECTED(
      publicResult, support.support.serverCall(lambda, args, evaluationKeys));
  return std::move(*publicResult);
}

MLIR_CAPI_EXPORTED std::string
library_get_shared_lib_path(LibrarySupport_Py support) {
  return support.support.getSharedLibPath();
}

MLIR_CAPI_EXPORTED std::string
library_get_client_parameters_path(LibrarySupport_Py support) {
  return support.support.getClientParametersPath();
}

// Client Support bindings ///////////////////////////////////////////////////

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
key_set(concretelang::clientlib::ClientParameters clientParameters,
        std::optional<concretelang::clientlib::KeySetCache> cache,
        uint64_t seedMsb, uint64_t seedLsb) {
  GET_OR_THROW_LLVM_EXPECTED(
      ks, (mlir::concretelang::LambdaSupport<int, int>::keySet(
              clientParameters, cache, seedMsb, seedLsb)));
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

MLIR_CAPI_EXPORTED std::vector<concretelang::clientlib::Data>
encrypt_arguments_new(
    concretelang::clientlib::ClientParameters clientParameters,
    concretelang::clientlib::KeySet &keySet,
    llvm::ArrayRef<mlir::concretelang::LambdaArgument *> args) {
  GET_OR_THROW_LLVM_EXPECTED(
      arguments,
      (mlir::concretelang::LambdaSupport<int, int>::exportArgumentsNew(
          clientParameters, keySet, args)));
  return std::move(*arguments);
}

MLIR_CAPI_EXPORTED lambdaArgument
decrypt_result(concretelang::clientlib::KeySet &keySet,
               concretelang::clientlib::PublicResult &publicResult) {
  GET_OR_THROW_LLVM_EXPECTED(
      result, mlir::concretelang::typedResult<
                  std::unique_ptr<mlir::concretelang::LambdaArgument>>(
                  keySet, publicResult));
  lambdaArgument result_{std::move(*result)};
  return result_;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
publicArgumentsUnserialize(
    mlir::concretelang::ClientParameters &clientParameters,
    const std::string &buffer) {
  std::stringstream istream(buffer);
  auto argsOrError = concretelang::clientlib::PublicArguments::unserialize(
      clientParameters, istream);
  if (!argsOrError) {
    throw std::runtime_error(argsOrError.error().mesg);
  }
  return std::move(argsOrError.value());
}

MLIR_CAPI_EXPORTED std::string publicArgumentsSerialize(
    concretelang::clientlib::PublicArguments &publicArguments) {

  std::ostringstream buffer(std::ios::binary);
  auto voidOrError = publicArguments.serialize(buffer);
  if (!voidOrError) {
    throw std::runtime_error(voidOrError.error().mesg);
  }
  return buffer.str();
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
publicResultUnserialize(mlir::concretelang::ClientParameters &clientParameters,
                        const std::string &buffer) {
  std::stringstream istream(buffer);
  auto publicResultOrError = concretelang::clientlib::PublicResult::unserialize(
      clientParameters, istream);
  if (!publicResultOrError) {
    throw std::runtime_error(publicResultOrError.error().mesg);
  }
  return std::move(publicResultOrError.value());
}

MLIR_CAPI_EXPORTED std::string
publicResultSerialize(concretelang::clientlib::PublicResult &publicResult) {
  std::ostringstream buffer(std::ios::binary);
  auto voidOrError = publicResult.serialize(buffer);
  if (!voidOrError) {
    throw std::runtime_error(voidOrError.error().mesg);
  }
  return buffer.str();
}

MLIR_CAPI_EXPORTED concretelang::clientlib::EvaluationKeys
evaluationKeysUnserialize(const std::string &buffer) {
  std::stringstream istream(buffer);

  concretelang::clientlib::EvaluationKeys evaluationKeys =
      concretelang::clientlib::readEvaluationKeys(istream);

  if (istream.fail()) {
    throw std::runtime_error("Cannot read evaluation keys");
  }

  return evaluationKeys;
}

MLIR_CAPI_EXPORTED std::string evaluationKeysSerialize(
    concretelang::clientlib::EvaluationKeys &evaluationKeys) {
  std::ostringstream buffer(std::ios::binary);
  concretelang::clientlib::operator<<(buffer, evaluationKeys);
  return buffer.str();
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
keySetUnserialize(const std::string &buffer) {
  std::stringstream istream(buffer);

  std::unique_ptr<concretelang::clientlib::KeySet> keySet =
      concretelang::clientlib::readKeySet(istream);

  if (istream.fail() || keySet.get() == nullptr) {
    throw std::runtime_error("Cannot read key set");
  }

  return keySet;
}

MLIR_CAPI_EXPORTED std::string
keySetSerialize(concretelang::clientlib::KeySet &keySet) {
  std::ostringstream buffer(std::ios::binary);
  concretelang::clientlib::operator<<(buffer, keySet);
  return buffer.str();
}

MLIR_CAPI_EXPORTED mlir::concretelang::ClientParameters
clientParametersUnserialize(const std::string &json) {
  GET_OR_THROW_LLVM_EXPECTED(
      clientParams,
      llvm::json::parse<mlir::concretelang::ClientParameters>(json));
  return clientParams.get();
}

MLIR_CAPI_EXPORTED std::string
clientParametersSerialize(mlir::concretelang::ClientParameters &params) {
  llvm::json::Value value(params);
  std::string jsonParams;
  llvm::raw_string_ostream buffer(jsonParams);
  buffer << value;
  return jsonParams;
}

MLIR_CAPI_EXPORTED void terminateDataflowParallelization() { _dfr_terminate(); }

MLIR_CAPI_EXPORTED void initDataflowParallelization() {
  mlir::concretelang::dfr::_dfr_set_required(true);
}

MLIR_CAPI_EXPORTED std::string roundTrip(const char *module) {
  std::shared_ptr<mlir::concretelang::CompilationContext> ccx =
      mlir::concretelang::CompilationContext::createShared();
  mlir::concretelang::CompilerEngine ce{ccx};

  std::string backingString;
  llvm::raw_string_ostream os(backingString);

  llvm::Expected<mlir::concretelang::CompilerEngine::CompilationResult>
      retOrErr = ce.compile(
          module, mlir::concretelang::CompilerEngine::Target::ROUND_TRIP);
  if (!retOrErr) {
    os << "MLIR parsing failed: " << llvm::toString(retOrErr.takeError());
    throw std::runtime_error(os.str());
  }

  retOrErr->mlirModuleRef->get().print(os);
  return os.str();
}

MLIR_CAPI_EXPORTED bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<uint8_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<uint16_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<uint32_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<uint64_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int8_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int16_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int32_t>>>() ||
         lambda_arg.ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int64_t>>>();
}

template <typename T, typename R>
MLIR_CAPI_EXPORTED std::vector<R> copyTensorLambdaArgumentTo64bitsvector(
    mlir::concretelang::TensorLambdaArgument<
        mlir::concretelang::IntLambdaArgument<T>> *tensor) {
  auto numElements = tensor->getNumElements();
  if (!numElements) {
    std::string backingString;
    llvm::raw_string_ostream os(backingString);
    os << "Couldn't get size of tensor: "
       << llvm::toString(std::move(numElements.takeError()));
    throw std::runtime_error(os.str());
  }
  std::vector<R> res;
  res.reserve(*numElements);
  T *data = tensor->getValue();
  for (size_t i = 0; i < *numElements; i++) {
    res.push_back(data[i]);
  }
  return res;
}

MLIR_CAPI_EXPORTED std::vector<uint64_t>
lambdaArgumentGetTensorData(lambdaArgument &lambda_arg) {
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint64_t>>>()) {
    llvm::Expected<size_t> sizeOrErr = arg->getNumElements();
    if (!sizeOrErr) {
      std::string backingString;
      llvm::raw_string_ostream os(backingString);
      os << "Couldn't get size of tensor: "
         << llvm::toString(sizeOrErr.takeError());
      throw std::runtime_error(os.str());
    }
    std::vector<uint64_t> data(arg->getValue(), arg->getValue() + *sizeOrErr);
    return data;
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint8_t>>>()) {
    return copyTensorLambdaArgumentTo64bitsvector<uint8_t, uint64_t>(arg);
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint16_t>>>()) {
    return copyTensorLambdaArgumentTo64bitsvector<uint16_t, uint64_t>(arg);
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint32_t>>>()) {
    return copyTensorLambdaArgumentTo64bitsvector<uint32_t, uint64_t>(arg);
  }
  throw std::invalid_argument(
      "LambdaArgument isn't a tensor or has an unsupported bitwidth");
}

MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetSignedTensorData(lambdaArgument &lambda_arg) {
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int64_t>>>()) {
    llvm::Expected<size_t> sizeOrErr = arg->getNumElements();
    if (!sizeOrErr) {
      std::string backingString;
      llvm::raw_string_ostream os(backingString);
      os << "Couldn't get size of tensor: "
         << llvm::toString(sizeOrErr.takeError());
      throw std::runtime_error(os.str());
    }
    std::vector<int64_t> data(arg->getValue(), arg->getValue() + *sizeOrErr);
    return data;
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int8_t>>>()) {
    return copyTensorLambdaArgumentTo64bitsvector<int8_t, int64_t>(arg);
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int16_t>>>()) {
    return copyTensorLambdaArgumentTo64bitsvector<int16_t, int64_t>(arg);
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int32_t>>>()) {
    return copyTensorLambdaArgumentTo64bitsvector<int32_t, int64_t>(arg);
  }
  throw std::invalid_argument(
      "LambdaArgument isn't a tensor or has an unsupported bitwidth");
}

MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg) {
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint8_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint16_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint32_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<uint64_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int8_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int16_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int32_t>>>()) {
    return arg->getDimensions();
  }
  if (auto arg =
          lambda_arg.ptr->dyn_cast<mlir::concretelang::TensorLambdaArgument<
              mlir::concretelang::IntLambdaArgument<int64_t>>>()) {
    return arg->getDimensions();
  }
  throw std::invalid_argument(
      "LambdaArgument isn't a tensor, should "
      "be a TensorLambdaArgument<IntLambdaArgument<(u)int{8,16,32,64}_t>>");
}

MLIR_CAPI_EXPORTED bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg) {
  auto ptr = lambda_arg.ptr;
  return ptr->isa<mlir::concretelang::IntLambdaArgument<uint64_t>>() ||
         ptr->isa<mlir::concretelang::IntLambdaArgument<int64_t>>();
}

MLIR_CAPI_EXPORTED bool lambdaArgumentIsSigned(lambdaArgument &lambda_arg) {
  auto ptr = lambda_arg.ptr;
  return ptr->isa<mlir::concretelang::IntLambdaArgument<int8_t>>() ||
         ptr->isa<mlir::concretelang::IntLambdaArgument<int16_t>>() ||
         ptr->isa<mlir::concretelang::IntLambdaArgument<int32_t>>() ||
         ptr->isa<mlir::concretelang::IntLambdaArgument<int64_t>>() ||
         ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int8_t>>>() ||
         ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int16_t>>>() ||
         ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int32_t>>>() ||
         ptr->isa<mlir::concretelang::TensorLambdaArgument<
             mlir::concretelang::IntLambdaArgument<int64_t>>>();
  ;
}

MLIR_CAPI_EXPORTED uint64_t
lambdaArgumentGetScalar(lambdaArgument &lambda_arg) {
  mlir::concretelang::IntLambdaArgument<uint64_t> *arg =
      lambda_arg.ptr
          ->dyn_cast<mlir::concretelang::IntLambdaArgument<uint64_t>>();
  if (arg == nullptr) {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<uint64_t>");
  }
  return arg->getValue();
}

MLIR_CAPI_EXPORTED int64_t
lambdaArgumentGetSignedScalar(lambdaArgument &lambda_arg) {
  mlir::concretelang::IntLambdaArgument<int64_t> *arg =
      lambda_arg.ptr
          ->dyn_cast<mlir::concretelang::IntLambdaArgument<int64_t>>();
  if (arg == nullptr) {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<int64_t>");
  }
  return arg->getValue();
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU8(
    std::vector<uint8_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint8_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI8(
    std::vector<int8_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<int8_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU16(
    std::vector<uint16_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint16_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI16(
    std::vector<int16_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<int16_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU32(
    std::vector<uint32_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint32_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI32(
    std::vector<int32_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<int32_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU64(
    std::vector<uint64_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<uint64_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI64(
    std::vector<int64_t> data, std::vector<int64_t> dimensions) {
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::TensorLambdaArgument<
          mlir::concretelang::IntLambdaArgument<int64_t>>>(data, dimensions)};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromScalar(uint64_t scalar) {
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::IntLambdaArgument<uint64_t>>(
          scalar)};
  return scalar_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument
lambdaArgumentFromSignedScalar(int64_t scalar) {
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::IntLambdaArgument<int64_t>>(scalar)};
  return scalar_arg;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::Data>
dataDeserialize(const std::string &buffer) {

  std::stringstream istream(buffer);

  auto deserializationResult =
      concretelang::clientlib::Data::deserialize(istream);
  if (!deserializationResult) {
    throw std::runtime_error(deserializationResult.error().mesg);
  }

  return std::move(deserializationResult.value());
}

MLIR_CAPI_EXPORTED std::string
dataSerialize(concretelang::clientlib::Data &data) {

  std::ostringstream buffer(std::ios::binary);

  auto serializationResult = data.serialize(buffer);
  if (!serializationResult) {
    throw std::runtime_error(serializationResult.error().mesg);
  }

  return buffer.str();
}

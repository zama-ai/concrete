// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Bindings/Python/CompilerAPIModule.h"
#include "concrete-optimizer.hpp"
#include "concrete-protocol.capnp.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Compat.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Dialect/FHE/IR/FHEOpsDialect.h.inc"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/GPUDFG.hpp"
#include "concretelang/ServerLib/ServerLib.h"
#include "concretelang/Support/logging.h"
#include <llvm/Support/Debug.h>
#include <mlir-c/Bindings/Python/Interop.h>
#include <mlir/CAPI/IR.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/ExecutionEngine/OptUtils.h>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <signal.h>
#include <stdexcept>
#include <string>

using mlir::concretelang::CompilationOptions;
using mlir::concretelang::LambdaArgument;

class SignalGuard {
public:
  SignalGuard() { previousHandler = signal(SIGINT, SignalGuard::handler); }
  ~SignalGuard() { signal(SIGINT, this->previousHandler); }

private:
  void (*previousHandler)(int);

  static void handler(int _signum) {
    llvm::outs() << " Aborting... \n";
    kill(getpid(), SIGKILL);
  }
};

/// Wrapper of the mlir::concretelang::LambdaArgument
struct lambdaArgument {
  std::shared_ptr<mlir::concretelang::LambdaArgument> ptr;
};
typedef struct lambdaArgument lambdaArgument;

/// Hold a list of lambdaArgument to represent execution arguments
struct executionArguments {
  lambdaArgument *data;
  size_t size;
};
typedef struct executionArguments executionArguments;

// Library Support bindings ///////////////////////////////////////////////////

struct LibrarySupport_Py {
  mlir::concretelang::LibrarySupport support;
};
typedef struct LibrarySupport_Py LibrarySupport_Py;

LibrarySupport_Py
library_support(const char *outputPath, const char *runtimeLibraryPath,
                bool generateSharedLib, bool generateStaticLib,
                bool generateClientParameters, bool generateCompilationFeedback,
                bool generateCppHeader) {
  return LibrarySupport_Py{mlir::concretelang::LibrarySupport(
      outputPath, runtimeLibraryPath, generateSharedLib, generateStaticLib,
      generateClientParameters, generateCompilationFeedback)};
}

std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibrarySupport_Py support, const char *module,
                mlir::concretelang::CompilationOptions options) {
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(module),
                        llvm::SMLoc());
  GET_OR_THROW_EXPECTED(auto compilationResult,
                        support.support.compile(sm, options));
  return compilationResult;
}

std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile_module(
    LibrarySupport_Py support, mlir::ModuleOp module,
    mlir::concretelang::CompilationOptions options,
    std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
  GET_OR_THROW_EXPECTED(auto compilationResult,
                        support.support.compile(module, cctx, options));
  return compilationResult;
}

concretelang::clientlib::ClientParameters library_load_client_parameters(
    LibrarySupport_Py support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_EXPECTED(auto clientParameters,
                        support.support.loadClientParameters(result));
  return clientParameters;
}

mlir::concretelang::ProgramCompilationFeedback
library_load_compilation_feedback(
    LibrarySupport_Py support,
    mlir::concretelang::LibraryCompilationResult &result) {
  GET_OR_THROW_EXPECTED(auto compilationFeedback,
                        support.support.loadCompilationFeedback(result));
  return compilationFeedback;
}

concretelang::serverlib::ServerLambda
library_load_server_lambda(LibrarySupport_Py support,
                           mlir::concretelang::LibraryCompilationResult &result,
                           std::string circuitName, bool useSimulation) {
  GET_OR_THROW_EXPECTED(
      auto serverLambda,
      support.support.loadServerLambda(result, circuitName, useSimulation));
  return serverLambda;
}

std::unique_ptr<concretelang::clientlib::PublicResult>
library_server_call(LibrarySupport_Py support,
                    concretelang::serverlib::ServerLambda lambda,
                    concretelang::clientlib::PublicArguments &args,
                    concretelang::clientlib::EvaluationKeys &evaluationKeys) {
  GET_OR_THROW_EXPECTED(auto publicResult, support.support.serverCall(
                                               lambda, args, evaluationKeys));
  return publicResult;
}

std::unique_ptr<concretelang::clientlib::PublicResult>
library_simulate(LibrarySupport_Py support,
                 concretelang::serverlib::ServerLambda lambda,
                 concretelang::clientlib::PublicArguments &args) {
  GET_OR_THROW_EXPECTED(auto publicResult,
                        support.support.simulate(lambda, args));
  return publicResult;
}

std::string library_get_shared_lib_path(LibrarySupport_Py support) {
  return support.support.getSharedLibPath();
}

std::string library_get_program_info_path(LibrarySupport_Py support) {
  return support.support.getProgramInfoPath();
}

// Client Support bindings ///////////////////////////////////////////////////

std::unique_ptr<concretelang::clientlib::KeySet>
key_set(concretelang::clientlib::ClientParameters clientParameters,
        std::optional<concretelang::clientlib::KeySetCache> cache,
        std::map<uint32_t, LweSecretKey> lweSecretKeys, uint64_t secretSeedMsb,
        uint64_t secretSeedLsb, uint64_t encSeedMsb, uint64_t encSeedLsb) {
  auto secretSeed = (((__uint128_t)secretSeedMsb) << 64) | secretSeedLsb;
  auto encryptionSeed = (((__uint128_t)encSeedMsb) << 64) | encSeedLsb;

  if (cache.has_value()) {
    GET_OR_THROW_RESULT(Keyset keyset,
                        (*cache).keysetCache.getKeyset(
                            clientParameters.programInfo.asReader().getKeyset(),
                            secretSeed, encryptionSeed, lweSecretKeys));
    concretelang::clientlib::KeySet output{keyset};
    return std::make_unique<concretelang::clientlib::KeySet>(std::move(output));
  } else {
    concretelang::csprng::SecretCSPRNG secCsprng(secretSeed);
    concretelang::csprng::EncryptionCSPRNG encCsprng(encryptionSeed);
    auto keyset = Keyset(clientParameters.programInfo.asReader().getKeyset(),
                         secCsprng, encCsprng, lweSecretKeys);
    concretelang::clientlib::KeySet output{keyset};
    return std::make_unique<concretelang::clientlib::KeySet>(std::move(output));
  }
}

std::unique_ptr<concretelang::clientlib::PublicArguments>
encrypt_arguments(concretelang::clientlib::ClientParameters clientParameters,
                  concretelang::clientlib::KeySet &keySet,
                  llvm::ArrayRef<mlir::concretelang::LambdaArgument *> args,
                  const std::string &circuitName) {
  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
          ::concretelang::csprng::EncryptionCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto circuit = maybeProgram.value().getClientCircuit(circuitName).value();
  std::vector<TransportValue> output;
  for (size_t i = 0; i < args.size(); i++) {
    auto info = circuit.getCircuitInfo().asReader().getInputs()[i];
    auto typeTransformer = getPythonTypeTransformer(info);
    auto input = typeTransformer(args[i]->value);
    auto maybePrepared = circuit.prepareInput(input, i);

    if (maybePrepared.has_failure()) {
      throw std::runtime_error(maybePrepared.as_failure().error().mesg);
    }
    output.push_back(maybePrepared.value());
  }
  concretelang::clientlib::PublicArguments publicArgs{output};
  return std::make_unique<concretelang::clientlib::PublicArguments>(
      std::move(publicArgs));
}

std::vector<lambdaArgument>
decrypt_result(concretelang::clientlib::ClientParameters clientParameters,
               concretelang::clientlib::KeySet &keySet,
               concretelang::clientlib::PublicResult &publicResult,
               const std::string &circuitName) {
  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
          ::concretelang::csprng::EncryptionCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto circuit = maybeProgram.value().getClientCircuit(circuitName).value();
  std::vector<lambdaArgument> results;
  for (auto e : llvm::enumerate(publicResult.values)) {
    auto maybeProcessed = circuit.processOutput(e.value(), e.index());
    if (maybeProcessed.has_failure()) {
      throw std::runtime_error(maybeProcessed.as_failure().error().mesg);
    }

    mlir::concretelang::LambdaArgument out{maybeProcessed.value()};
    lambdaArgument tensor_arg{
        std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
    results.push_back(tensor_arg);
  }
  return results;
}

std::unique_ptr<concretelang::clientlib::PublicArguments>
publicArgumentsUnserialize(
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &buffer) {
  auto publicArgumentsProto = Message<concreteprotocol::PublicArguments>();
  if (publicArgumentsProto.readBinaryFromString(buffer).has_failure()) {
    throw std::runtime_error("Failed to deserialize public arguments.");
  }
  std::vector<TransportValue> values;
  for (auto arg : publicArgumentsProto.asReader().getArgs()) {
    values.push_back(arg);
  }
  concretelang::clientlib::PublicArguments output{values};
  return std::make_unique<concretelang::clientlib::PublicArguments>(
      std::move(output));
}

std::string publicArgumentsSerialize(
    concretelang::clientlib::PublicArguments &publicArguments) {
  auto publicArgumentsProto = Message<concreteprotocol::PublicArguments>();
  auto argBuilder =
      publicArgumentsProto.asBuilder().initArgs(publicArguments.values.size());
  for (size_t i = 0; i < publicArguments.values.size(); i++) {
    argBuilder.setWithCaveats(i, publicArguments.values[i].asReader());
  }
  auto maybeBuffer = publicArgumentsProto.writeBinaryToString();
  if (maybeBuffer.has_failure()) {
    throw std::runtime_error("Failed to serialize public arguments.");
  }
  return maybeBuffer.value();
}

std::unique_ptr<concretelang::clientlib::PublicResult> publicResultUnserialize(
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &buffer) {
  auto publicResultsProto = Message<concreteprotocol::PublicResults>();
  if (publicResultsProto.readBinaryFromString(buffer).has_failure()) {
    throw std::runtime_error("Failed to deserialize public results.");
  }
  std::vector<TransportValue> values;
  for (auto res : publicResultsProto.asReader().getResults()) {
    values.push_back(res);
  }
  concretelang::clientlib::PublicResult output{values};
  return std::make_unique<concretelang::clientlib::PublicResult>(
      std::move(output));
}

std::string
publicResultSerialize(concretelang::clientlib::PublicResult &publicResult) {
  std::string buffer;
  auto publicResultsProto = Message<concreteprotocol::PublicResults>();
  auto resBuilder =
      publicResultsProto.asBuilder().initResults(publicResult.values.size());
  for (size_t i = 0; i < publicResult.values.size(); i++) {
    resBuilder.setWithCaveats(i, publicResult.values[i].asReader());
  }
  auto maybeBuffer = publicResultsProto.writeBinaryToString();
  if (maybeBuffer.has_failure()) {
    throw std::runtime_error("Failed to serialize public results.");
  }
  return maybeBuffer.value();
}

concretelang::clientlib::EvaluationKeys
evaluationKeysUnserialize(const std::string &buffer) {
  auto serverKeysetProto = Message<concreteprotocol::ServerKeyset>();
  auto maybeError = serverKeysetProto.readBinaryFromString(
      buffer, mlir::concretelang::python::DESER_OPTIONS);
  if (maybeError.has_failure()) {
    throw std::runtime_error("Failed to deserialize server keyset." +
                             maybeError.as_failure().error().mesg);
  }
  auto serverKeyset =
      concretelang::keysets::ServerKeyset::fromProto(serverKeysetProto);
  concretelang::clientlib::EvaluationKeys output{serverKeyset};
  return output;
}

std::string evaluationKeysSerialize(
    concretelang::clientlib::EvaluationKeys &evaluationKeys) {
  auto serverKeysetProto = evaluationKeys.keyset.toProto();
  auto maybeBuffer = serverKeysetProto.writeBinaryToString();
  if (maybeBuffer.has_failure()) {
    throw std::runtime_error("Failed to serialize evaluation keys.");
  }
  return maybeBuffer.value();
}

std::unique_ptr<concretelang::clientlib::KeySet>
keySetUnserialize(const std::string &buffer) {
  auto keysetProto = Message<concreteprotocol::Keyset>();
  auto maybeError = keysetProto.readBinaryFromString(
      buffer, mlir::concretelang::python::DESER_OPTIONS);
  if (maybeError.has_failure()) {
    throw std::runtime_error("Failed to deserialize keyset." +
                             maybeError.as_failure().error().mesg);
  }
  auto keyset = concretelang::keysets::Keyset::fromProto(keysetProto);
  concretelang::clientlib::KeySet output{keyset};
  return std::make_unique<concretelang::clientlib::KeySet>(std::move(output));
}

std::string keySetSerialize(concretelang::clientlib::KeySet &keySet) {
  auto keysetProto = keySet.keyset.toProto();
  auto maybeBuffer = keysetProto.writeBinaryToString();
  if (maybeBuffer.has_failure()) {
    throw std::runtime_error("Failed to serialize keys.");
  }
  return maybeBuffer.value();
}

concretelang::clientlib::SharedScalarOrTensorData
valueUnserialize(const std::string &buffer) {
  auto inner = TransportValue();
  if (inner
          .readBinaryFromString(buffer,
                                mlir::concretelang::python::DESER_OPTIONS)
          .has_failure()) {
    throw std::runtime_error("Failed to deserialize Value");
  }
  return {inner};
}

std::string
valueSerialize(const concretelang::clientlib::SharedScalarOrTensorData &value) {
  auto maybeString = value.value.writeBinaryToString();
  if (maybeString.has_failure()) {
    throw std::runtime_error("Failed to serialize Value");
  }
  return maybeString.value();
}

concretelang::clientlib::ValueExporter
createValueExporter(concretelang::clientlib::KeySet &keySet,
                    concretelang::clientlib::ClientParameters &clientParameters,
                    const std::string &circuitName) {
  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
          ::concretelang::csprng::EncryptionCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(circuitName);
  return ::concretelang::clientlib::ValueExporter{maybeCircuit.value()};
}

concretelang::clientlib::SimulatedValueExporter createSimulatedValueExporter(
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &circuitName) {

  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo, ::concretelang::keysets::ClientKeyset(),
      std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
          ::concretelang::csprng::EncryptionCSPRNG(0)),
      true);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(circuitName);
  return ::concretelang::clientlib::SimulatedValueExporter{
      maybeCircuit.value()};
}

concretelang::clientlib::ValueDecrypter createValueDecrypter(
    concretelang::clientlib::KeySet &keySet,
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &circuitName) {

  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
          ::concretelang::csprng::EncryptionCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(circuitName);
  return ::concretelang::clientlib::ValueDecrypter{maybeCircuit.value()};
}

concretelang::clientlib::SimulatedValueDecrypter createSimulatedValueDecrypter(
    concretelang::clientlib::ClientParameters &clientParameters,
    const std::string &circuitName) {

  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(),
      ::concretelang::keysets::ClientKeyset(),
      std::make_shared<::concretelang::csprng::EncryptionCSPRNG>(
          ::concretelang::csprng::EncryptionCSPRNG(0)),
      true);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(circuitName);
  return ::concretelang::clientlib::SimulatedValueDecrypter{
      maybeCircuit.value()};
}

concretelang::clientlib::ClientParameters
clientParametersUnserialize(const std::string &json) {
  auto programInfo = Message<concreteprotocol::ProgramInfo>();
  if (programInfo.readJsonFromString(json).has_failure()) {
    throw std::runtime_error("Failed to deserialize client parameters");
  }
  return concretelang::clientlib::ClientParameters{programInfo, {}, {}, {}, {}};
}

std::string
clientParametersSerialize(concretelang::clientlib::ClientParameters &params) {
  auto maybeJson = params.programInfo.writeJsonToString();
  if (maybeJson.has_failure()) {
    throw std::runtime_error("Failed to serialize client parameters");
  }
  return maybeJson.value();
}

void terminateDataflowParallelization() { _dfr_terminate(); }

void initDataflowParallelization() {
  mlir::concretelang::dfr::_dfr_set_required(true);
}

bool checkGPURuntimeEnabled() {
  return mlir::concretelang::gpu_dfg::check_cuda_runtime_enabled();
}

bool checkCudaDeviceAvailable() {
  return mlir::concretelang::gpu_dfg::check_cuda_device_available();
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
    os << "MLIR parsing failed: " << llvm::toString(retOrErr.takeError());
    throw std::runtime_error(os.str());
  }

  retOrErr->mlirModuleRef->get().print(os);
  return os.str();
}

bool lambdaArgumentIsTensor(lambdaArgument &lambda_arg) {
  return !lambda_arg.ptr->value.isScalar();
}

std::vector<uint64_t> lambdaArgumentGetTensorData(lambdaArgument &lambda_arg) {
  if (auto tensor = lambda_arg.ptr->value.getTensor<uint8_t>(); tensor) {
    Tensor<uint64_t> out = (Tensor<uint64_t>)tensor.value();
    return out.values;
  } else if (auto tensor = lambda_arg.ptr->value.getTensor<uint16_t>();
             tensor) {
    Tensor<uint64_t> out = (Tensor<uint64_t>)tensor.value();
    return out.values;
  } else if (auto tensor = lambda_arg.ptr->value.getTensor<uint32_t>();
             tensor) {
    Tensor<uint64_t> out = (Tensor<uint64_t>)tensor.value();
    return out.values;
  } else if (auto tensor = lambda_arg.ptr->value.getTensor<uint64_t>();
             tensor) {
    return tensor.value().values;
  } else {
    throw std::invalid_argument(
        "LambdaArgument isn't a tensor or has an unsupported bitwidth");
  }
}

std::vector<int64_t>
lambdaArgumentGetSignedTensorData(lambdaArgument &lambda_arg) {
  if (auto tensor = lambda_arg.ptr->value.getTensor<int8_t>(); tensor) {
    Tensor<int64_t> out = (Tensor<int64_t>)tensor.value();
    return out.values;
  } else if (auto tensor = lambda_arg.ptr->value.getTensor<int16_t>(); tensor) {
    Tensor<int64_t> out = (Tensor<int64_t>)tensor.value();
    return out.values;
  } else if (auto tensor = lambda_arg.ptr->value.getTensor<int32_t>(); tensor) {
    Tensor<int64_t> out = (Tensor<int64_t>)tensor.value();
    return out.values;
  } else if (auto tensor = lambda_arg.ptr->value.getTensor<int64_t>(); tensor) {
    return tensor.value().values;
  } else {
    throw std::invalid_argument(
        "LambdaArgument isn't a tensor or has an unsupported bitwidth");
  }
}

std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg) {
  std::vector<size_t> dims = lambda_arg.ptr->value.getDimensions();
  return {dims.begin(), dims.end()};
}

bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->value.isScalar();
}

bool lambdaArgumentIsSigned(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->value.isSigned();
}

uint64_t lambdaArgumentGetScalar(lambdaArgument &lambda_arg) {
  if (lambda_arg.ptr->value.isScalar() &&
      lambda_arg.ptr->value.hasElementType<uint64_t>()) {
    return lambda_arg.ptr->value.getTensor<uint64_t>()->values[0];
  } else {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<uint64_t>");
  }
}

int64_t lambdaArgumentGetSignedScalar(lambdaArgument &lambda_arg) {
  if (lambda_arg.ptr->value.isScalar() &&
      lambda_arg.ptr->value.hasElementType<int64_t>()) {
    return lambda_arg.ptr->value.getTensor<int64_t>()->values[0];
  } else {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<int64_t>");
  }
}

lambdaArgument lambdaArgumentFromTensorU8(std::vector<uint8_t> data,
                                          std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());

  auto val = Value{((Tensor<int64_t>)Tensor<uint8_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorI8(std::vector<int8_t> data,
                                          std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int8_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorU16(std::vector<uint16_t> data,
                                           std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<uint16_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorI16(std::vector<int16_t> data,
                                           std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int16_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorU32(std::vector<uint32_t> data,
                                           std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<uint32_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorI32(std::vector<int32_t> data,
                                           std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int32_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorU64(std::vector<uint64_t> data,
                                           std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<uint64_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromTensorI64(std::vector<int64_t> data,
                                           std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int64_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

lambdaArgument lambdaArgumentFromScalar(uint64_t scalar) {
  auto val = Value{((Tensor<int64_t>)Tensor<uint64_t>(scalar))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return scalar_arg;
}

lambdaArgument lambdaArgumentFromSignedScalar(int64_t scalar) {
  auto val = Value{Tensor<int64_t>(scalar)};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return scalar_arg;
}

concreteprotocol::LweCiphertextEncryptionInfo::Reader
clientParametersInputEncryptionAt(
    ::concretelang::clientlib::ClientParameters &clientParameters,
    size_t inputIdx, std::string circuitName) {
  auto reader = clientParameters.programInfo.asReader();
  if (!reader.hasCircuits()) {
    throw std::runtime_error("can't get keyid: no circuit info");
  }
  auto circuits = reader.getCircuits();
  for (auto circuit : circuits) {
    if (circuit.hasName() &&
        circuitName.compare(circuit.getName().cStr()) == 0) {
      if (!circuit.hasInputs()) {
        throw std::runtime_error("can't get keyid: no input");
      }
      auto inputs = circuit.getInputs();
      if (inputIdx >= inputs.size()) {
        throw std::runtime_error(
            "can't get keyid: inputIdx bigger than number of inputs");
      }
      auto input = inputs[inputIdx];
      if (!input.hasTypeInfo()) {
        throw std::runtime_error("can't get keyid: input don't have typeInfo");
      }
      auto typeInfo = input.getTypeInfo();
      if (!typeInfo.hasLweCiphertext()) {
        throw std::runtime_error("can't get keyid: typeInfo don't "
                                 "have lwe ciphertext info");
      }
      auto lweCt = typeInfo.getLweCiphertext();
      if (!lweCt.hasEncryption()) {
        throw std::runtime_error("can't get keyid: lwe ciphertext "
                                 "don't have encryption info");
      }
      return lweCt.getEncryption();
    }
  }

  throw std::runtime_error("can't get keyid: no circuit with name " +
                           circuitName);
}

/// Populate the compiler API python module.
void mlir::concretelang::python::populateCompilerAPISubmodule(
    pybind11::module &m) {
  m.doc() = "Concretelang compiler python API";

  m.def("round_trip",
        [](std::string mlir_input) { return roundTrip(mlir_input.c_str()); });

  m.def("set_llvm_debug_flag", [](bool enable) { llvm::DebugFlag = enable; });

  m.def("set_compiler_logging",
        [](bool enable) { mlir::concretelang::setupLogging(enable); });

  m.def("terminate_df_parallelization", &terminateDataflowParallelization);

  m.def("init_df_parallelization", &initDataflowParallelization);
  m.def("check_gpu_runtime_enabled", &checkGPURuntimeEnabled);
  m.def("check_cuda_device_available", &checkCudaDeviceAvailable);

  m.def("import_tfhers_fheuint8",
        [](const pybind11::bytes &serialized_fheuint,
           TfhersFheIntDescription info, uint32_t encryptionKeyId,
           double encryptionVariance) {
          const std::string &buffer_str = serialized_fheuint;
          std::vector<uint8_t> buffer(buffer_str.begin(), buffer_str.end());
          auto arrayRef = llvm::ArrayRef<uint8_t>(buffer);
          auto valueOrError = ::concretelang::clientlib::importTfhersFheUint8(
              arrayRef, info, encryptionKeyId, encryptionVariance);
          if (valueOrError.has_error()) {
            throw std::runtime_error(valueOrError.error().mesg);
          }
          return ::concretelang::clientlib::SharedScalarOrTensorData{
              valueOrError.value()};
        });

  m.def("export_tfhers_fheuint8",
        [](::concretelang::clientlib::SharedScalarOrTensorData fheuint,
           TfhersFheIntDescription info) {
          auto result = ::concretelang::clientlib::exportTfhersFheUint8(
              fheuint.value, info);
          if (result.has_error()) {
            throw std::runtime_error(result.error().mesg);
          }
          return result.value();
        });

  m.def("get_tfhers_fheuint8_description",
        [](const pybind11::bytes &serialized_fheuint) {
          const std::string &buffer_str = serialized_fheuint;
          std::vector<uint8_t> buffer(buffer_str.begin(), buffer_str.end());
          auto arrayRef = llvm::ArrayRef<uint8_t>(buffer);
          auto info =
              ::concretelang::clientlib::getTfhersFheUint8Description(arrayRef);
          if (info.has_error()) {
            throw std::runtime_error(info.error().mesg);
          }
          return info.value();
        });

  pybind11::class_<TfhersFheIntDescription>(m, "TfhersFheIntDescription")
      .def(pybind11::init([](size_t width, bool is_signed,
                             size_t message_modulus, size_t carry_modulus,
                             size_t degree, size_t lwe_size, size_t n_cts,
                             size_t noise_level, bool ks_first) {
        auto desc = TfhersFheIntDescription();
        desc.width = width;
        desc.is_signed = is_signed;
        desc.message_modulus = message_modulus;
        desc.carry_modulus = carry_modulus;
        desc.degree = degree;
        desc.lwe_size = lwe_size;
        desc.n_cts = n_cts;
        desc.noise_level = noise_level;
        desc.ks_first = ks_first;
        return desc;
      }))
      .def_static("UNKNOWN_NOISE_LEVEL",
                  [] { return concrete_cpu_tfhers_unknown_noise_level(); })
      .def_property(
          "width", [](TfhersFheIntDescription &desc) { return desc.width; },
          [](TfhersFheIntDescription &desc, size_t width) {
            desc.width = width;
          })
      .def_property(
          "message_modulus",
          [](TfhersFheIntDescription &desc) { return desc.message_modulus; },
          [](TfhersFheIntDescription &desc, size_t message_modulus) {
            desc.message_modulus = message_modulus;
          })
      .def_property(
          "carry_modulus",
          [](TfhersFheIntDescription &desc) { return desc.carry_modulus; },
          [](TfhersFheIntDescription &desc, size_t carry_modulus) {
            desc.carry_modulus = carry_modulus;
          })
      .def_property(
          "degree", [](TfhersFheIntDescription &desc) { return desc.degree; },
          [](TfhersFheIntDescription &desc, size_t degree) {
            desc.degree = degree;
          })
      .def_property(
          "lwe_size",
          [](TfhersFheIntDescription &desc) { return desc.lwe_size; },
          [](TfhersFheIntDescription &desc, size_t lwe_size) {
            desc.lwe_size = lwe_size;
          })
      .def_property(
          "n_cts", [](TfhersFheIntDescription &desc) { return desc.n_cts; },
          [](TfhersFheIntDescription &desc, size_t n_cts) {
            desc.n_cts = n_cts;
          })
      .def_property(
          "noise_level",
          [](TfhersFheIntDescription &desc) { return desc.noise_level; },
          [](TfhersFheIntDescription &desc, size_t noise_level) {
            desc.noise_level = noise_level;
          })
      .def_property(
          "is_signed",
          [](TfhersFheIntDescription &desc) { return desc.is_signed; },
          [](TfhersFheIntDescription &desc, bool is_signed) {
            desc.is_signed = is_signed;
          })
      .def_property(
          "ks_first",
          [](TfhersFheIntDescription &desc) { return desc.ks_first; },
          [](TfhersFheIntDescription &desc, bool ks_first) {
            desc.ks_first = ks_first;
          });

  pybind11::enum_<mlir::concretelang::Backend>(m, "Backend")
      .value("CPU", mlir::concretelang::Backend::CPU)
      .value("GPU", mlir::concretelang::Backend::GPU)
      .export_values();

  pybind11::enum_<optimizer::Strategy>(m, "OptimizerStrategy")
      .value("V0", optimizer::Strategy::V0)
      .value("DAG_MONO", optimizer::Strategy::DAG_MONO)
      .value("DAG_MULTI", optimizer::Strategy::DAG_MULTI)
      .export_values();

  pybind11::enum_<concrete_optimizer::MultiParamStrategy>(
      m, "OptimizerMultiParameterStrategy")
      .value("PRECISION", concrete_optimizer::MultiParamStrategy::ByPrecision)
      .value("PRECISION_AND_NORM2",
             concrete_optimizer::MultiParamStrategy::ByPrecisionAndNorm2)
      .export_values();

  pybind11::enum_<concrete_optimizer::Encoding>(m, "Encoding")
      .value("AUTO", concrete_optimizer::Encoding::Auto)
      .value("CRT", concrete_optimizer::Encoding::Crt)
      .value("NATIVE", concrete_optimizer::Encoding::Native)
      .export_values();

  pybind11::class_<CompilationOptions>(m, "CompilationOptions")
      .def(pybind11::init([](mlir::concretelang::Backend backend) {
        return CompilationOptions(backend);
      }))
      .def("set_verify_diagnostics",
           [](CompilationOptions &options, bool b) {
             options.verifyDiagnostics = b;
           })
      .def("set_auto_parallelize", [](CompilationOptions &options,
                                      bool b) { options.autoParallelize = b; })
      .def("set_loop_parallelize", [](CompilationOptions &options,
                                      bool b) { options.loopParallelize = b; })
      .def("set_dataflow_parallelize",
           [](CompilationOptions &options, bool b) {
             options.dataflowParallelize = b;
           })
      .def("set_compress_evaluation_keys",
           [](CompilationOptions &options, bool b) {
             options.compressEvaluationKeys = b;
           })
      .def("set_compress_input_ciphertexts",
           [](CompilationOptions &options, bool b) {
             options.compressInputCiphertexts = b;
           })
      .def("set_optimize_concrete", [](CompilationOptions &options,
                                       bool b) { options.optimizeTFHE = b; })
      .def("set_p_error",
           [](CompilationOptions &options, double p_error) {
             options.optimizerConfig.p_error = p_error;
           })
      .def("set_display_optimizer_choice",
           [](CompilationOptions &options, bool display) {
             options.optimizerConfig.display = display;
           })
      .def("set_optimizer_strategy",
           [](CompilationOptions &options, optimizer::Strategy strategy) {
             options.optimizerConfig.strategy = strategy;
           })
      .def("set_optimizer_multi_parameter_strategy",
           [](CompilationOptions &options,
              concrete_optimizer::MultiParamStrategy strategy) {
             options.optimizerConfig.multi_param_strategy = strategy;
           })
      .def("set_global_p_error",
           [](CompilationOptions &options, double global_p_error) {
             options.optimizerConfig.global_p_error = global_p_error;
           })
      .def("add_composition",
           [](CompilationOptions &options, std::string from_func,
              size_t from_pos, std::string to_func, size_t to_pos) {
             options.optimizerConfig.composition_rules.push_back(
                 {from_func, from_pos, to_func, to_pos});
           })
      .def("set_composable",
           [](CompilationOptions &options, bool composable) {
             options.optimizerConfig.composable = composable;
           })
      .def("set_security_level",
           [](CompilationOptions &options, int security_level) {
             options.optimizerConfig.security = security_level;
           })
      .def("set_v0_parameter",
           [](CompilationOptions &options, size_t glweDimension,
              size_t logPolynomialSize, size_t nSmall, size_t brLevel,
              size_t brLogBase, size_t ksLevel, size_t ksLogBase) {
             options.v0Parameter = {glweDimension, logPolynomialSize, nSmall,
                                    brLevel,       brLogBase,         ksLevel,
                                    ksLogBase,     std::nullopt};
           })
      .def("set_v0_parameter",
           [](CompilationOptions &options, size_t glweDimension,
              size_t logPolynomialSize, size_t nSmall, size_t brLevel,
              size_t brLogBase, size_t ksLevel, size_t ksLogBase,
              mlir::concretelang::CRTDecomposition crtDecomposition,
              size_t cbsLevel, size_t cbsLogBase, size_t pksLevel,
              size_t pksLogBase, size_t pksInputLweDimension,
              size_t pksOutputPolynomialSize) {
             mlir::concretelang::PackingKeySwitchParameter pksParam = {
                 pksInputLweDimension, pksOutputPolynomialSize, pksLevel,
                 pksLogBase};
             mlir::concretelang::CitcuitBoostrapParameter crbParam = {
                 cbsLevel, cbsLogBase};
             mlir::concretelang::WopPBSParameter wopPBSParam = {pksParam,
                                                                crbParam};
             mlir::concretelang::LargeIntegerParameter largeIntegerParam = {
                 crtDecomposition, wopPBSParam};
             options.v0Parameter = {glweDimension, logPolynomialSize, nSmall,
                                    brLevel,       brLogBase,         ksLevel,
                                    ksLogBase,     largeIntegerParam};
           })
      .def("force_encoding",
           [](CompilationOptions &options,
              concrete_optimizer::Encoding encoding) {
             options.optimizerConfig.encoding = encoding;
           })
      .def("simulation", [](CompilationOptions &options,
                            bool simulate) { options.simulate = simulate; })
      .def("set_emit_gpu_ops",
           [](CompilationOptions &options, bool emit_gpu_ops) {
             options.emitGPUOps = emit_gpu_ops;
           })
      .def("set_batch_tfhe_ops",
           [](CompilationOptions &options, bool batch_tfhe_ops) {
             options.batchTFHEOps = batch_tfhe_ops;
           })
      .def("set_enable_tlu_fusing",
           [](CompilationOptions &options, bool enableTluFusing) {
             options.enableTluFusing = enableTluFusing;
           })
      .def("set_print_tlu_fusing",
           [](CompilationOptions &options, bool printTluFusing) {
             options.printTluFusing = printTluFusing;
           })
      .def("set_enable_overflow_detection_in_simulation",
           [](CompilationOptions &options, bool enableOverflowDetection) {
             options.enableOverflowDetectionInSimulation =
                 enableOverflowDetection;
           });

  pybind11::enum_<mlir::concretelang::PrimitiveOperation>(m,
                                                          "PrimitiveOperation")
      .value("PBS", mlir::concretelang::PrimitiveOperation::PBS)
      .value("WOP_PBS", mlir::concretelang::PrimitiveOperation::WOP_PBS)
      .value("KEY_SWITCH", mlir::concretelang::PrimitiveOperation::KEY_SWITCH)
      .value("CLEAR_ADDITION",
             mlir::concretelang::PrimitiveOperation::CLEAR_ADDITION)
      .value("ENCRYPTED_ADDITION",
             mlir::concretelang::PrimitiveOperation::ENCRYPTED_ADDITION)
      .value("CLEAR_MULTIPLICATION",
             mlir::concretelang::PrimitiveOperation::CLEAR_MULTIPLICATION)
      .value("ENCRYPTED_NEGATION",
             mlir::concretelang::PrimitiveOperation::ENCRYPTED_NEGATION)
      .export_values();

  pybind11::enum_<mlir::concretelang::KeyType>(m, "KeyType")
      .value("SECRET", mlir::concretelang::KeyType::SECRET)
      .value("BOOTSTRAP", mlir::concretelang::KeyType::BOOTSTRAP)
      .value("KEY_SWITCH", mlir::concretelang::KeyType::KEY_SWITCH)
      .value("PACKING_KEY_SWITCH",
             mlir::concretelang::KeyType::PACKING_KEY_SWITCH)
      .export_values();

  pybind11::class_<mlir::concretelang::Statistic>(m, "Statistic")
      .def_readonly("operation", &mlir::concretelang::Statistic::operation)
      .def_readonly("location", &mlir::concretelang::Statistic::location)
      .def_readonly("keys", &mlir::concretelang::Statistic::keys)
      .def_readonly("count", &mlir::concretelang::Statistic::count);

  pybind11::class_<mlir::concretelang::ProgramCompilationFeedback>(
      m, "ProgramCompilationFeedback")
      .def_readonly("complexity",
                    &mlir::concretelang::ProgramCompilationFeedback::complexity)
      .def_readonly("p_error",
                    &mlir::concretelang::ProgramCompilationFeedback::pError)
      .def_readonly(
          "global_p_error",
          &mlir::concretelang::ProgramCompilationFeedback::globalPError)
      .def_readonly(
          "total_secret_keys_size",
          &mlir::concretelang::ProgramCompilationFeedback::totalSecretKeysSize)
      .def_readonly("total_bootstrap_keys_size",
                    &mlir::concretelang::ProgramCompilationFeedback::
                        totalBootstrapKeysSize)
      .def_readonly("total_keyswitch_keys_size",
                    &mlir::concretelang::ProgramCompilationFeedback::
                        totalKeyswitchKeysSize)
      .def_readonly(
          "circuit_feedbacks",
          &mlir::concretelang::ProgramCompilationFeedback::circuitFeedbacks);

  pybind11::class_<mlir::concretelang::CircuitCompilationFeedback>(
      m, "CircuitCompilationFeedback")
      .def_readonly("name",
                    &mlir::concretelang::CircuitCompilationFeedback::name)
      .def_readonly(
          "total_inputs_size",
          &mlir::concretelang::CircuitCompilationFeedback::totalInputsSize)
      .def_readonly(
          "total_output_size",
          &mlir::concretelang::CircuitCompilationFeedback::totalOutputsSize)
      .def_readonly("crt_decompositions_of_outputs",
                    &mlir::concretelang::CircuitCompilationFeedback::
                        crtDecompositionsOfOutputs)
      .def_readonly("statistics",
                    &mlir::concretelang::CircuitCompilationFeedback::statistics)
      .def_readonly(
          "memory_usage_per_location",
          &mlir::concretelang::CircuitCompilationFeedback::memoryUsagePerLoc);

  pybind11::class_<mlir::concretelang::CompilationContext,
                   std::shared_ptr<mlir::concretelang::CompilationContext>>(
      m, "CompilationContext")
      .def(pybind11::init([]() {
        return mlir::concretelang::CompilationContext::createShared();
      }))
      .def("mlir_context",
           [](std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
             auto mlirCtx = cctx->getMLIRContext();
             return pybind11::reinterpret_steal<pybind11::object>(
                 mlirPythonContextToCapsule(wrap(mlirCtx)));
           });

  pybind11::class_<mlir::concretelang::LibraryCompilationResult>(
      m, "LibraryCompilationResult")
      .def(pybind11::init([](std::string outputDirPath) {
        return mlir::concretelang::LibraryCompilationResult{outputDirPath};
      }));
  pybind11::class_<::concretelang::serverlib::ServerLambda>(m, "LibraryLambda");
  pybind11::class_<LibrarySupport_Py>(m, "LibrarySupport")
      .def(pybind11::init(
          [](std::string outputPath, std::string runtimeLibraryPath,
             bool generateSharedLib, bool generateStaticLib,
             bool generateClientParameters, bool generateCompilationFeedback,
             bool generateCppHeader) {
            return library_support(
                outputPath.c_str(), runtimeLibraryPath.c_str(),
                generateSharedLib, generateStaticLib, generateClientParameters,
                generateCompilationFeedback, generateCppHeader);
          }))
      .def("compile",
           [](LibrarySupport_Py &support, std::string mlir_program,
              mlir::concretelang::CompilationOptions options) {
             SignalGuard signalGuard;
             return library_compile(support, mlir_program.c_str(), options);
           })
      .def("compile",
           [](LibrarySupport_Py &support, pybind11::object mlir_module,
              mlir::concretelang::CompilationOptions options,
              std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
             SignalGuard signalGuard;
             return library_compile_module(
                 support,
                 unwrap(mlirPythonCapsuleToModule(mlir_module.ptr())).clone(),
                 options, cctx);
           })
      .def("load_client_parameters",
           [](LibrarySupport_Py &support,
              mlir::concretelang::LibraryCompilationResult &result) {
             return library_load_client_parameters(support, result);
           })
      .def("load_compilation_feedback",
           [](LibrarySupport_Py &support,
              mlir::concretelang::LibraryCompilationResult &result) {
             return library_load_compilation_feedback(support, result);
           })
      .def(
          "load_server_lambda",
          [](LibrarySupport_Py &support,
             mlir::concretelang::LibraryCompilationResult &result,
             std::string circuitName, bool useSimulation) {
            return library_load_server_lambda(support, result, circuitName,
                                              useSimulation);
          },
          pybind11::return_value_policy::reference)
      .def("server_call",
           [](LibrarySupport_Py &support,
              ::concretelang::serverlib::ServerLambda lambda,
              ::concretelang::clientlib::PublicArguments &publicArguments,
              ::concretelang::clientlib::EvaluationKeys &evaluationKeys) {
             pybind11::gil_scoped_release release;
             SignalGuard signalGuard;
             return library_server_call(support, lambda, publicArguments,
                                        evaluationKeys);
           })
      .def("simulate",
           [](LibrarySupport_Py &support,
              ::concretelang::serverlib::ServerLambda lambda,
              ::concretelang::clientlib::PublicArguments &publicArguments) {
             pybind11::gil_scoped_release release;
             return library_simulate(support, lambda, publicArguments);
           })
      .def("get_shared_lib_path",
           [](LibrarySupport_Py &support) {
             return library_get_shared_lib_path(support);
           })
      .def("get_program_info_path", [](LibrarySupport_Py &support) {
        return library_get_program_info_path(support);
      });

  class ClientSupport {};
  pybind11::class_<ClientSupport>(m, "ClientSupport")
      .def(pybind11::init())
      .def_static(
          "key_set",
          [](::concretelang::clientlib::ClientParameters clientParameters,
             ::concretelang::clientlib::KeySetCache *cache,
             uint64_t secretSeedMsb, uint64_t secretSeedLsb,
             uint64_t encSeedMsb, uint64_t encSeedLsb,
             std::map<uint32_t, LweSecretKey> initialLweSecretKeys) {
            SignalGuard signalGuard;
            auto optCache =
                cache == nullptr
                    ? std::nullopt
                    : std::optional<::concretelang::clientlib::KeySetCache>(
                          *cache);
            return key_set(clientParameters, optCache, initialLweSecretKeys,
                           secretSeedMsb, secretSeedLsb, encSeedMsb,
                           encSeedLsb);
          },
          pybind11::arg().none(false), pybind11::arg().none(true),
          pybind11::arg("secretSeedMsb") = 0,
          pybind11::arg("secretSeedLsb") = 0, pybind11::arg("encSeedMsb") = 0,
          pybind11::arg("encSeedLsb") = 0,
          pybind11::arg("initialLweSecretKeys") =
              std::map<uint32_t, LweSecretKey>())
      .def_static(
          "encrypt_arguments",
          [](::concretelang::clientlib::ClientParameters clientParameters,
             ::concretelang::clientlib::KeySet &keySet,
             std::vector<lambdaArgument> args, const std::string &circuitName) {
            std::vector<mlir::concretelang::LambdaArgument *> argsRef;
            for (auto i = 0u; i < args.size(); i++) {
              argsRef.push_back(args[i].ptr.get());
            }
            return encrypt_arguments(clientParameters, keySet, argsRef,
                                     circuitName);
          })
      .def_static(
          "decrypt_result",
          [](::concretelang::clientlib::ClientParameters clientParameters,
             ::concretelang::clientlib::KeySet &keySet,
             ::concretelang::clientlib::PublicResult &publicResult,
             const std::string &circuitName) {
            return decrypt_result(clientParameters, keySet, publicResult,
                                  circuitName);
          });
  pybind11::class_<::concretelang::clientlib::KeySetCache>(m, "KeySetCache")
      .def(pybind11::init<std::string &>());

  pybind11::class_<::concretelang::clientlib::LweSecretKeyParam>(
      m, "LweSecretKeyParam")
      .def("dimension", [](::concretelang::clientlib::LweSecretKeyParam &key) {
        return key.info.asReader().getParams().getLweDimension();
      });

  pybind11::class_<::concretelang::clientlib::BootstrapKeyParam>(
      m, "BootstrapKeyParam")
      .def("input_secret_key_id",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getInputId();
           })
      .def("output_secret_key_id",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getOutputId();
           })
      .def("level",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getLevelCount();
           })
      .def("base_log",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getBaseLog();
           })
      .def("glwe_dimension",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getGlweDimension();
           })
      .def("variance",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getVariance();
           })
      .def("polynomial_size",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getPolynomialSize();
           })
      .def("input_lwe_dimension",
           [](::concretelang::clientlib::BootstrapKeyParam &key) {
             return key.info.asReader().getParams().getInputLweDimension();
           });

  pybind11::class_<::concretelang::clientlib::KeyswitchKeyParam>(
      m, "KeyswitchKeyParam")
      .def("input_secret_key_id",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getInputId();
           })
      .def("output_secret_key_id",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getOutputId();
           })
      .def("level",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getLevelCount();
           })
      .def("base_log",
           [](::concretelang::clientlib::KeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getBaseLog();
           })
      .def("variance", [](::concretelang::clientlib::KeyswitchKeyParam &key) {
        return key.info.asReader().getParams().getVariance();
      });

  pybind11::class_<::concretelang::clientlib::PackingKeyswitchKeyParam>(
      m, "PackingKeyswitchKeyParam")
      .def("input_secret_key_id",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getInputId();
           })
      .def("output_secret_key_id",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getOutputId();
           })
      .def("level",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getLevelCount();
           })
      .def("base_log",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getBaseLog();
           })
      .def("glwe_dimension",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getGlweDimension();
           })
      .def("polynomial_size",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getPolynomialSize();
           })
      .def("input_lwe_dimension",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getInputLweDimension();
           })
      .def("variance",
           [](::concretelang::clientlib::PackingKeyswitchKeyParam &key) {
             return key.info.asReader().getParams().getVariance();
           });

  pybind11::class_<::concretelang::clientlib::ClientParameters>(
      m, "ClientParameters")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return clientParametersUnserialize(buffer);
                  })
      .def("serialize",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             return pybind11::bytes(
                 clientParametersSerialize(clientParameters));
           })
      .def("lwe_secret_key_param",
           [](::concretelang::clientlib::ClientParameters &clientParameters,
              size_t keyId) {
             if (keyId >= clientParameters.secretKeys.size()) {
               throw std::runtime_error("keyId bigger than the number of keys");
             }
             return clientParameters.secretKeys[keyId];
           })
      .def("input_keyid_at",
           [](::concretelang::clientlib::ClientParameters &clientParameters,
              size_t inputIdx, std::string circuitName) {
             auto encryption = clientParametersInputEncryptionAt(
                 clientParameters, inputIdx, circuitName);
             return encryption.getKeyId();
           })
      .def("input_variance_at",
           [](::concretelang::clientlib::ClientParameters &clientParameters,
              size_t inputIdx, std::string circuitName) {
             auto encryption = clientParametersInputEncryptionAt(
                 clientParameters, inputIdx, circuitName);
             return encryption.getVariance();
           })
      .def("function_list",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             std::vector<std::string> result;
             for (auto circuit :
                  clientParameters.programInfo.asReader().getCircuits()) {
               result.push_back(circuit.getName());
             }
             return result;
           })
      .def("output_signs",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             std::vector<bool> result;
             for (auto output : clientParameters.programInfo.asReader()
                                    .getCircuits()[0]
                                    .getOutputs()) {
               if (output.getTypeInfo().hasLweCiphertext() &&
                   output.getTypeInfo()
                       .getLweCiphertext()
                       .getEncoding()
                       .hasInteger()) {
                 result.push_back(output.getTypeInfo()
                                      .getLweCiphertext()
                                      .getEncoding()
                                      .getInteger()
                                      .getIsSigned());
               } else {
                 result.push_back(true);
               }
             }
             return result;
           })
      .def("input_signs",
           [](::concretelang::clientlib::ClientParameters &clientParameters) {
             std::vector<bool> result;
             for (auto input : clientParameters.programInfo.asReader()
                                   .getCircuits()[0]
                                   .getInputs()) {
               if (input.getTypeInfo().hasLweCiphertext() &&
                   input.getTypeInfo()
                       .getLweCiphertext()
                       .getEncoding()
                       .hasInteger()) {
                 result.push_back(input.getTypeInfo()
                                      .getLweCiphertext()
                                      .getEncoding()
                                      .getInteger()
                                      .getIsSigned());
               } else {
                 result.push_back(true);
               }
             }
             return result;
           })
      .def_readonly("secret_keys",
                    &::concretelang::clientlib::ClientParameters::secretKeys)
      .def_readonly("bootstrap_keys",
                    &::concretelang::clientlib::ClientParameters::bootstrapKeys)
      .def_readonly("keyswitch_keys",
                    &::concretelang::clientlib::ClientParameters::keyswitchKeys)
      .def_readonly(
          "packing_keyswitch_keys",
          &::concretelang::clientlib::ClientParameters::packingKeyswitchKeys);

  pybind11::class_<LweSecretKey>(m, "LweSecretKey")
      .def_static(
          "deserialize",
          [](pybind11::bytes buffer,
             ::concretelang::clientlib::LweSecretKeyParam &params) {
            std::string buffer_str = buffer;
            auto lwe_dim = params.info.asReader().getParams().getLweDimension();
            auto lwe_size = concrete_cpu_lwe_secret_key_size_u64(lwe_dim);
            std::vector<uint64_t> lwe_sk(lwe_size);
            auto key_size = concrete_cpu_unserialize_lwe_secret_key_u64(
                (uint8_t *)buffer_str.data(), buffer_str.size(), lwe_sk.data(),
                lwe_sk.size());
            lwe_sk.resize(key_size);
            return LweSecretKey(
                std::make_shared<std::vector<uint64_t>>(std::move(lwe_sk)),
                params.info);
          })
      .def("serialize",
           [](LweSecretKey &lweSk) {
             auto skBuffer = lweSk.getBuffer();
             auto lwe_dimension =
                 lweSk.getInfo().asReader().getParams().getLweDimension();
             auto buffer_size =
                 concrete_cpu_lwe_secret_key_buffer_size_u64(lwe_dimension);
             std::vector<uint8_t> buffer(buffer_size, 0);
             buffer_size = concrete_cpu_serialize_lwe_secret_key_u64(
                 skBuffer.data(), lwe_dimension, buffer.data(), buffer_size);
             if (buffer_size == 0) {
               throw std::runtime_error("couldn't serialize the secret key");
             }
             auto bytes = pybind11::bytes((char *)buffer.data(), buffer_size);
             return bytes;
           })
      .def_static(
          "deserialize_from_glwe",
          [](pybind11::bytes buffer,
             ::concretelang::clientlib::LweSecretKeyParam &params) {
            std::string buffer_str = buffer;
            auto lwe_dim = params.info.asReader().getParams().getLweDimension();
            auto glwe_sk_size = concrete_cpu_lwe_secret_key_size_u64(lwe_dim);
            std::vector<uint64_t> glwe_sk(glwe_sk_size);
            auto key_size = concrete_cpu_unserialize_glwe_secret_key_u64(
                (uint8_t *)buffer_str.data(), buffer_str.size(), glwe_sk.data(),
                glwe_sk.size());
            glwe_sk.resize(key_size);
            return LweSecretKey(
                std::make_shared<std::vector<uint64_t>>(std::move(glwe_sk)),
                params.info);
          })
      .def("serialize_as_glwe",
           [](LweSecretKey &lweSk, size_t glwe_dimension,
              size_t polynomial_size) {
             auto skBuffer = lweSk.getBuffer();
             auto buffer_size = concrete_cpu_glwe_secret_key_buffer_size_u64(
                 glwe_dimension, polynomial_size);
             std::vector<uint8_t> buffer(buffer_size, 0);
             buffer_size = concrete_cpu_serialize_glwe_secret_key_u64(
                 skBuffer.data(), glwe_dimension, polynomial_size,
                 buffer.data(), buffer_size);
             if (buffer_size == 0) {
               throw std::runtime_error("couldn't serialize the secret key");
             }
             auto bytes = pybind11::bytes((char *)buffer.data(), buffer_size);
             return bytes;
           })
      .def_property_readonly("param", [](LweSecretKey &lweSk) {
        return ::concretelang::clientlib::LweSecretKeyParam{lweSk.getInfo()};
      });

  pybind11::class_<::concretelang::clientlib::KeySet>(m, "KeySet")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    std::unique_ptr<::concretelang::clientlib::KeySet> result =
                        keySetUnserialize(buffer);
                    return result;
                  })
      .def("serialize",
           [](::concretelang::clientlib::KeySet &keySet) {
             return pybind11::bytes(keySetSerialize(keySet));
           })
      .def("get_lwe_secret_key",
           [](::concretelang::clientlib::KeySet &keySet, size_t keyIndex) {
             auto secretKeys = keySet.keyset.client.lweSecretKeys;
             if (keyIndex >= secretKeys.size()) {
               throw std::runtime_error(
                   "keyIndex is bigger than the number of keys");
             }
             return secretKeys[keyIndex];
           })
      .def("get_evaluation_keys",
           [](::concretelang::clientlib::KeySet &keySet) {
             return ::concretelang::clientlib::EvaluationKeys{
                 keySet.keyset.server};
           });

  pybind11::class_<::concretelang::clientlib::SharedScalarOrTensorData>(m,
                                                                        "Value")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return valueUnserialize(buffer);
                  })
      .def(
          "serialize",
          [](const ::concretelang::clientlib::SharedScalarOrTensorData &value) {
            return pybind11::bytes(valueSerialize(value));
          });

  pybind11::class_<ServerProgram>(m, "ServerProgram")
      .def_static("load",
                  [](LibrarySupport_Py &support, bool useSimulation) {
                    GET_OR_THROW_EXPECTED(auto programInfo,
                                          support.support.getProgramInfo());
                    auto sharedLibPath = support.support.getSharedLibPath();
                    GET_OR_THROW_RESULT(
                        auto result,
                        ServerProgram::load(programInfo.asReader(),
                                            sharedLibPath, useSimulation));
                    return result;
                  })
      .def("get_server_circuit", [](ServerProgram &program,
                                    const std::string &circuitName) {
        GET_OR_THROW_RESULT(auto result, program.getServerCircuit(circuitName));
        return result;
      });

  pybind11::class_<ServerCircuit>(m, "ServerCircuit")
      .def("call",
           [](ServerCircuit &circuit,
              ::concretelang::clientlib::PublicArguments &publicArguments,
              ::concretelang::clientlib::EvaluationKeys &evaluationKeys) {
             SignalGuard signalGuard;
             pybind11::gil_scoped_release release;
             auto keyset = evaluationKeys.keyset;
             auto values = publicArguments.values;
             GET_OR_THROW_RESULT(auto output, circuit.call(keyset, values));
             ::concretelang::clientlib::PublicResult res{output};
             return std::make_unique<::concretelang::clientlib::PublicResult>(
                 std::move(res));
           })
      .def("simulate",
           [](ServerCircuit &circuit,
              ::concretelang::clientlib::PublicArguments &publicArguments) {
             pybind11::gil_scoped_release release;
             auto values = publicArguments.values;
             GET_OR_THROW_RESULT(auto output, circuit.simulate(values));
             ::concretelang::clientlib::PublicResult res{output};
             return std::make_unique<::concretelang::clientlib::PublicResult>(
                 std::move(res));
           });

  pybind11::class_<::concretelang::clientlib::ValueExporter>(m, "ValueExporter")
      .def_static(
          "create",
          [](::concretelang::clientlib::KeySet &keySet,
             ::concretelang::clientlib::ClientParameters &clientParameters,
             const std::string &circuitName) {
            return createValueExporter(keySet, clientParameters, circuitName);
          })
      .def("export_scalar",
           [](::concretelang::clientlib::ValueExporter &exporter,
              size_t position, int64_t value) {
             SignalGuard signalGuard;
             pybind11::gil_scoped_release release;

             auto info = exporter.circuit.getCircuitInfo()
                             .asReader()
                             .getInputs()[position];
             auto typeTransformer = getPythonTypeTransformer(info);
             auto result = exporter.circuit.prepareInput(
                 typeTransformer({Tensor<int64_t>(value)}), position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return ::concretelang::clientlib::SharedScalarOrTensorData{
                 result.value()};
           })
      .def("export_tensor", [](::concretelang::clientlib::ValueExporter
                                   &exporter,
                               size_t position, std::vector<int64_t> values,
                               std::vector<int64_t> shape) {
        SignalGuard signalGuard;
        pybind11::gil_scoped_release release;
        std::vector<size_t> dimensions(shape.begin(), shape.end());
        auto info =
            exporter.circuit.getCircuitInfo().asReader().getInputs()[position];
        auto typeTransformer = getPythonTypeTransformer(info);
        auto result = exporter.circuit.prepareInput(
            typeTransformer({Tensor<int64_t>(values, dimensions)}), position);

        if (result.has_error()) {
          throw std::runtime_error(result.error().mesg);
        }

        return ::concretelang::clientlib::SharedScalarOrTensorData{
            result.value()};
      });

  pybind11::class_<::concretelang::clientlib::SimulatedValueExporter>(
      m, "SimulatedValueExporter")
      .def_static(
          "create",
          [](::concretelang::clientlib::ClientParameters &clientParameters,
             const std::string &circuitName) {
            return createSimulatedValueExporter(clientParameters, circuitName);
          })
      .def("export_scalar",
           [](::concretelang::clientlib::SimulatedValueExporter &exporter,
              size_t position, int64_t value) {
             SignalGuard signalGuard;
             auto info = exporter.circuit.getCircuitInfo()
                             .asReader()
                             .getInputs()[position];
             auto typeTransformer = getPythonTypeTransformer(info);
             auto result = exporter.circuit.prepareInput(
                 typeTransformer({Tensor<int64_t>(value)}), position);

             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return ::concretelang::clientlib::SharedScalarOrTensorData{
                 result.value()};
           })
      .def("export_tensor", [](::concretelang::clientlib::SimulatedValueExporter
                                   &exporter,
                               size_t position, std::vector<int64_t> values,
                               std::vector<int64_t> shape) {
        SignalGuard signalGuard;
        std::vector<size_t> dimensions(shape.begin(), shape.end());
        auto info =
            exporter.circuit.getCircuitInfo().asReader().getInputs()[position];
        auto typeTransformer = getPythonTypeTransformer(info);
        auto result = exporter.circuit.prepareInput(
            typeTransformer({Tensor<int64_t>(values, dimensions)}), position);

        if (result.has_error()) {
          throw std::runtime_error(result.error().mesg);
        }

        return ::concretelang::clientlib::SharedScalarOrTensorData{
            result.value()};
      });

  pybind11::class_<::concretelang::clientlib::ValueDecrypter>(m,
                                                              "ValueDecrypter")
      .def_static(
          "create",
          [](::concretelang::clientlib::KeySet &keySet,
             ::concretelang::clientlib::ClientParameters &clientParameters,
             const std::string &circuitName) {
            return createValueDecrypter(keySet, clientParameters, circuitName);
          })
      .def("decrypt",
           [](::concretelang::clientlib::ValueDecrypter &decrypter,
              size_t position,
              ::concretelang::clientlib::SharedScalarOrTensorData &value) {
             SignalGuard signalGuard;
             pybind11::gil_scoped_release release;

             auto result =
                 decrypter.circuit.processOutput(value.value, position);
             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return lambdaArgument{
                 std::make_shared<mlir::concretelang::LambdaArgument>(
                     mlir::concretelang::LambdaArgument{result.value()})};
           });

  pybind11::class_<::concretelang::clientlib::SimulatedValueDecrypter>(
      m, "SimulatedValueDecrypter")
      .def_static(
          "create",
          [](::concretelang::clientlib::ClientParameters &clientParameters,
             const std::string &circuitName) {
            return createSimulatedValueDecrypter(clientParameters, circuitName);
          })
      .def("decrypt",
           [](::concretelang::clientlib::SimulatedValueDecrypter &decrypter,
              size_t position,
              ::concretelang::clientlib::SharedScalarOrTensorData &value) {
             SignalGuard signalGuard;

             auto result =
                 decrypter.circuit.processOutput(value.value, position);
             if (result.has_error()) {
               throw std::runtime_error(result.error().mesg);
             }

             return lambdaArgument{
                 std::make_shared<mlir::concretelang::LambdaArgument>(
                     mlir::concretelang::LambdaArgument{result.value()})};
           });

  pybind11::class_<::concretelang::clientlib::PublicArguments,
                   std::unique_ptr<::concretelang::clientlib::PublicArguments>>(
      m, "PublicArguments")
      .def_static(
          "create",
          [](const ::concretelang::clientlib::ClientParameters
                 &clientParameters,
             std::vector<::concretelang::clientlib::SharedScalarOrTensorData>
                 &buffers) {
            std::vector<TransportValue> vals;
            for (auto buf : buffers) {
              vals.push_back(buf.value);
            }
            return ::concretelang::clientlib::PublicArguments{vals};
          })
      .def_static(
          "deserialize",
          [](::concretelang::clientlib::ClientParameters &clientParameters,
             const pybind11::bytes &buffer) {
            return publicArgumentsUnserialize(clientParameters, buffer);
          })
      .def("serialize",
           [](::concretelang::clientlib::PublicArguments &publicArgument) {
             return pybind11::bytes(publicArgumentsSerialize(publicArgument));
           });
  pybind11::class_<::concretelang::clientlib::PublicResult>(m, "PublicResult")
      .def_static(
          "deserialize",
          [](::concretelang::clientlib::ClientParameters &clientParameters,
             const pybind11::bytes &buffer) {
            return publicResultUnserialize(clientParameters, buffer);
          })
      .def("serialize",
           [](::concretelang::clientlib::PublicResult &publicResult) {
             return pybind11::bytes(publicResultSerialize(publicResult));
           })
      .def("n_values",
           [](const ::concretelang::clientlib::PublicResult &publicResult) {
             return publicResult.values.size();
           })
      .def("get_value",
           [](::concretelang::clientlib::PublicResult &publicResult,
              size_t position) {
             if (position >= publicResult.values.size()) {
               throw std::runtime_error("Failed to get public result value.");
             }
             return ::concretelang::clientlib::SharedScalarOrTensorData{
                 publicResult.values[position]};
           });

  pybind11::class_<::concretelang::clientlib::EvaluationKeys>(m,
                                                              "EvaluationKeys")
      .def_static("deserialize",
                  [](const pybind11::bytes &buffer) {
                    return evaluationKeysUnserialize(buffer);
                  })
      .def("serialize",
           [](::concretelang::clientlib::EvaluationKeys &evaluationKeys) {
             return pybind11::bytes(evaluationKeysSerialize(evaluationKeys));
           });

  pybind11::class_<lambdaArgument>(m, "LambdaArgument")
      .def_static("from_tensor_u8",
                  [](std::vector<uint8_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU8(tensor, dims);
                  })
      .def_static("from_tensor_u16",
                  [](std::vector<uint16_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU16(tensor, dims);
                  })
      .def_static("from_tensor_u32",
                  [](std::vector<uint32_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU32(tensor, dims);
                  })
      .def_static("from_tensor_u64",
                  [](std::vector<uint64_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorU64(tensor, dims);
                  })
      .def_static("from_tensor_i8",
                  [](std::vector<int8_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI8(tensor, dims);
                  })
      .def_static("from_tensor_i16",
                  [](std::vector<int16_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI16(tensor, dims);
                  })
      .def_static("from_tensor_i32",
                  [](std::vector<int32_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI32(tensor, dims);
                  })
      .def_static("from_tensor_i64",
                  [](std::vector<int64_t> tensor, std::vector<int64_t> dims) {
                    return lambdaArgumentFromTensorI64(tensor, dims);
                  })
      .def_static("from_scalar", lambdaArgumentFromScalar)
      .def_static("from_signed_scalar", lambdaArgumentFromSignedScalar)
      .def("is_tensor",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsTensor(lambda_arg);
           })
      .def("get_tensor_data",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetTensorData(lambda_arg);
           })
      .def("get_signed_tensor_data",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetSignedTensorData(lambda_arg);
           })
      .def("get_tensor_shape",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetTensorDimensions(lambda_arg);
           })
      .def("is_scalar",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsScalar(lambda_arg);
           })
      .def("is_signed",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentIsSigned(lambda_arg);
           })
      .def("get_scalar",
           [](lambdaArgument &lambda_arg) {
             return lambdaArgumentGetScalar(lambda_arg);
           })
      .def("get_signed_scalar", [](lambdaArgument &lambda_arg) {
        return lambdaArgumentGetSignedScalar(lambda_arg);
      });
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "llvm/ADT/SmallString.h"
#include <cstdint>
#include <memory>
#include <stdexcept>

#include "concrete-protocol.capnp.h"
#include "concretelang/Bindings/Python/CompilerEngine.h"
#include "concretelang/Common/Compat.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Support/CompilerEngine.h"

// Library Support bindings ///////////////////////////////////////////////////
MLIR_CAPI_EXPORTED LibrarySupport_Py
library_support(const char *outputPath, const char *runtimeLibraryPath,
                bool generateSharedLib, bool generateStaticLib,
                bool generateClientParameters, bool generateCompilationFeedback,
                bool generateCppHeader) {
  return LibrarySupport_Py{mlir::concretelang::LibrarySupport(
      outputPath, runtimeLibraryPath, generateSharedLib, generateStaticLib,
      generateClientParameters, generateCompilationFeedback)};
}

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile(LibrarySupport_Py support, const char *module,
                mlir::concretelang::CompilationOptions options) {
  llvm::SourceMgr sm;
  sm.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(module),
                        llvm::SMLoc());
  GET_OR_THROW_LLVM_EXPECTED(compilationResult,
                             support.support.compile(sm, options));
  return std::move(*compilationResult);
}

MLIR_CAPI_EXPORTED std::unique_ptr<mlir::concretelang::LibraryCompilationResult>
library_compile_module(
    LibrarySupport_Py support, mlir::ModuleOp module,
    mlir::concretelang::CompilationOptions options,
    std::shared_ptr<mlir::concretelang::CompilationContext> cctx) {
  GET_OR_THROW_LLVM_EXPECTED(compilationResult,
                             support.support.compile(module, cctx, options));
  return std::move(*compilationResult);
}

MLIR_CAPI_EXPORTED concretelang::clientlib::ClientParameters
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
library_load_server_lambda(LibrarySupport_Py support,
                           mlir::concretelang::LibraryCompilationResult &result,
                           bool useSimulation) {
  GET_OR_THROW_LLVM_EXPECTED(
      serverLambda, support.support.loadServerLambda(result, useSimulation));
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

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
library_simulate(LibrarySupport_Py support,
                 concretelang::serverlib::ServerLambda lambda,
                 concretelang::clientlib::PublicArguments &args) {
  GET_OR_THROW_LLVM_EXPECTED(publicResult,
                             support.support.simulate(lambda, args));
  return std::move(*publicResult);
}

MLIR_CAPI_EXPORTED std::string
library_get_shared_lib_path(LibrarySupport_Py support) {
  return support.support.getSharedLibPath();
}

MLIR_CAPI_EXPORTED std::string
library_get_program_info_path(LibrarySupport_Py support) {
  return support.support.getProgramInfoPath();
}

// Client Support bindings ///////////////////////////////////////////////////

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::KeySet>
key_set(concretelang::clientlib::ClientParameters clientParameters,
        std::optional<concretelang::clientlib::KeySetCache> cache,
        uint64_t seedMsb, uint64_t seedLsb) {
  if (cache.has_value()) {
    GET_OR_THROW_RESULT(Keyset keyset,
                        (*cache).keysetCache.getKeyset(
                            clientParameters.programInfo.asReader().getKeyset(),
                            seedMsb, seedLsb));
    concretelang::clientlib::KeySet output{keyset,
                                           clientParameters.programInfo};
    return std::make_unique<concretelang::clientlib::KeySet>(std::move(output));
  } else {
    __uint128_t seed = seedMsb;
    seed <<= 64;
    seed += seedLsb;
    auto csprng = concretelang::csprng::ConcreteCSPRNG(seed);
    auto keyset =
        Keyset(clientParameters.programInfo.asReader().getKeyset(), csprng);
    concretelang::clientlib::KeySet output{keyset,
                                           clientParameters.programInfo};
    return std::make_unique<concretelang::clientlib::KeySet>(std::move(output));
  }
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
encrypt_arguments(concretelang::clientlib::ClientParameters clientParameters,
                  concretelang::clientlib::KeySet &keySet,
                  llvm::ArrayRef<mlir::concretelang::LambdaArgument *> args) {
  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<CSPRNG>(::concretelang::csprng::ConcreteCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto circuit = maybeProgram.value()
                     .getClientCircuit(clientParameters.programInfo.asReader()
                                           .getCircuits()[0]
                                           .getName())
                     .value();
  std::vector<TransportValue> output;
  for (size_t i = 0; i < args.size(); i++) {
    auto info =
        clientParameters.programInfo.asReader().getCircuits()[0].getInputs()[i];
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

MLIR_CAPI_EXPORTED lambdaArgument
decrypt_result(concretelang::clientlib::ClientParameters clientParameters,
               concretelang::clientlib::KeySet &keySet,
               concretelang::clientlib::PublicResult &publicResult) {
  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<CSPRNG>(::concretelang::csprng::ConcreteCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  if (publicResult.values.size() != 1) {
    throw std::runtime_error("Tried to decrypt with wrong arity.");
  }
  auto circuit = maybeProgram.value()
                     .getClientCircuit(clientParameters.programInfo.asReader()
                                           .getCircuits()[0]
                                           .getName())
                     .value();
  auto maybeProcessed = circuit.processOutput(publicResult.values[0], 0);
  if (maybeProcessed.has_failure()) {
    throw std::runtime_error(maybeProcessed.as_failure().error().mesg);
  }

  mlir::concretelang::LambdaArgument out{maybeProcessed.value()};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicArguments>
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

MLIR_CAPI_EXPORTED std::string publicArgumentsSerialize(
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

MLIR_CAPI_EXPORTED std::unique_ptr<concretelang::clientlib::PublicResult>
publicResultUnserialize(
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

MLIR_CAPI_EXPORTED std::string
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

MLIR_CAPI_EXPORTED concretelang::clientlib::SharedScalarOrTensorData
valueUnserialize(const std::string &buffer) {
  auto inner = TransportValue();
  if (inner.readBinaryFromString(buffer).has_failure()) {
    throw std::runtime_error("Failed to deserialize Value");
  }
  return {inner};
}

MLIR_CAPI_EXPORTED std::string
valueSerialize(const concretelang::clientlib::SharedScalarOrTensorData &value) {
  auto maybeString = value.value.writeBinaryToString();
  if (maybeString.has_failure()) {
    throw std::runtime_error("Failed to serialize Value");
  }
  return maybeString.value();
}

MLIR_CAPI_EXPORTED concretelang::clientlib::ValueExporter createValueExporter(
    concretelang::clientlib::KeySet &keySet,
    concretelang::clientlib::ClientParameters &clientParameters) {
  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<CSPRNG>(::concretelang::csprng::ConcreteCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(
      clientParameters.programInfo.asReader().getCircuits()[0].getName());
  return ::concretelang::clientlib::ValueExporter{maybeCircuit.value()};
}

MLIR_CAPI_EXPORTED concretelang::clientlib::SimulatedValueExporter
createSimulatedValueExporter(
    concretelang::clientlib::ClientParameters &clientParameters) {

  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo, ::concretelang::keysets::ClientKeyset(),
      std::make_shared<CSPRNG>(::concretelang::csprng::ConcreteCSPRNG(0)),
      true);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(
      clientParameters.programInfo.asReader().getCircuits()[0].getName());
  return ::concretelang::clientlib::SimulatedValueExporter{
      maybeCircuit.value()};
}

MLIR_CAPI_EXPORTED concretelang::clientlib::ValueDecrypter createValueDecrypter(
    concretelang::clientlib::KeySet &keySet,
    concretelang::clientlib::ClientParameters &clientParameters) {

  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(), keySet.keyset.client,
      std::make_shared<CSPRNG>(::concretelang::csprng::ConcreteCSPRNG(0)),
      false);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(
      clientParameters.programInfo.asReader().getCircuits()[0].getName());
  return ::concretelang::clientlib::ValueDecrypter{maybeCircuit.value()};
}

MLIR_CAPI_EXPORTED concretelang::clientlib::SimulatedValueDecrypter
createSimulatedValueDecrypter(
    concretelang::clientlib::ClientParameters &clientParameters) {

  auto maybeProgram = ::concretelang::clientlib::ClientProgram::create(
      clientParameters.programInfo.asReader(),
      ::concretelang::keysets::ClientKeyset(),
      std::make_shared<CSPRNG>(::concretelang::csprng::ConcreteCSPRNG(0)),
      true);
  if (maybeProgram.has_failure()) {
    throw std::runtime_error(maybeProgram.as_failure().error().mesg);
  }
  auto maybeCircuit = maybeProgram.value().getClientCircuit(
      clientParameters.programInfo.asReader().getCircuits()[0].getName());
  return ::concretelang::clientlib::SimulatedValueDecrypter{
      maybeCircuit.value()};
}

MLIR_CAPI_EXPORTED concretelang::clientlib::ClientParameters
clientParametersUnserialize(const std::string &json) {
  auto programInfo = Message<concreteprotocol::ProgramInfo>();
  if (programInfo.readJsonFromString(json).has_failure()) {
    throw std::runtime_error("Failed to deserialize client parameters");
  }
  return concretelang::clientlib::ClientParameters{programInfo, {}, {}, {}, {}};
}

MLIR_CAPI_EXPORTED std::string
clientParametersSerialize(concretelang::clientlib::ClientParameters &params) {
  auto maybeJson = params.programInfo.writeJsonToString();
  if (maybeJson.has_failure()) {
    throw std::runtime_error("Failed to serialize client parameters");
  }
  return maybeJson.value();
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
  return !lambda_arg.ptr->value.isScalar();
}

MLIR_CAPI_EXPORTED std::vector<uint64_t>
lambdaArgumentGetTensorData(lambdaArgument &lambda_arg) {
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

MLIR_CAPI_EXPORTED std::vector<int64_t>
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

MLIR_CAPI_EXPORTED std::vector<int64_t>
lambdaArgumentGetTensorDimensions(lambdaArgument &lambda_arg) {
  std::vector<size_t> dims = lambda_arg.ptr->value.getDimensions();
  return {dims.begin(), dims.end()};
}

MLIR_CAPI_EXPORTED bool lambdaArgumentIsScalar(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->value.isScalar();
}

MLIR_CAPI_EXPORTED bool lambdaArgumentIsSigned(lambdaArgument &lambda_arg) {
  return lambda_arg.ptr->value.isSigned();
}

MLIR_CAPI_EXPORTED uint64_t
lambdaArgumentGetScalar(lambdaArgument &lambda_arg) {
  if (lambda_arg.ptr->value.isScalar() &&
      lambda_arg.ptr->value.hasElementType<uint64_t>()) {
    return lambda_arg.ptr->value.getTensor<uint64_t>()->values[0];
  } else {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<uint64_t>");
  }
}

MLIR_CAPI_EXPORTED int64_t
lambdaArgumentGetSignedScalar(lambdaArgument &lambda_arg) {
  if (lambda_arg.ptr->value.isScalar() &&
      lambda_arg.ptr->value.hasElementType<int64_t>()) {
    return lambda_arg.ptr->value.getTensor<int64_t>()->values[0];
  } else {
    throw std::invalid_argument("LambdaArgument isn't a scalar, should "
                                "be an IntLambdaArgument<int64_t>");
  }
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU8(
    std::vector<uint8_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());

  auto val = Value{((Tensor<int64_t>)Tensor<uint8_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI8(
    std::vector<int8_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int8_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU16(
    std::vector<uint16_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<uint16_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI16(
    std::vector<int16_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int16_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU32(
    std::vector<uint32_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<uint32_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI32(
    std::vector<int32_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int32_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorU64(
    std::vector<uint64_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<uint64_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromTensorI64(
    std::vector<int64_t> data, std::vector<int64_t> dimensions) {
  std::vector<size_t> dims(dimensions.begin(), dimensions.end());
  auto val = Value{((Tensor<int64_t>)Tensor<int64_t>(data, dims))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument tensor_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return tensor_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument lambdaArgumentFromScalar(uint64_t scalar) {
  auto val = Value{((Tensor<int64_t>)Tensor<uint64_t>(scalar))};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return scalar_arg;
}

MLIR_CAPI_EXPORTED lambdaArgument
lambdaArgumentFromSignedScalar(int64_t scalar) {
  auto val = Value{Tensor<int64_t>(scalar)};
  mlir::concretelang::LambdaArgument out{val};
  lambdaArgument scalar_arg{
      std::make_shared<mlir::concretelang::LambdaArgument>(std::move(out))};
  return scalar_arg;
}

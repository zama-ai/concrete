// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <optional>
#include <string>
#include <variant>

#include "boost/outcome.h"
#include "concrete-cpu.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Transformers.h"
#include "concretelang/Common/Values.h"

using concretelang::error::Result;
using concretelang::keysets::ClientKeyset;
using concretelang::transformers::InputTransformer;
using concretelang::transformers::OutputTransformer;
using concretelang::transformers::TransformerFactory;
using concretelang::values::TransportValue;
using concretelang::values::Value;

namespace concretelang {
namespace clientlib {

bool ClientCircuit::isSimulated() { return simulated; }

Result<ClientCircuit>
ClientCircuit::create(const Message<concreteprotocol::CircuitInfo> &info,
                      const ClientKeyset &keyset,
                      std::shared_ptr<csprng::EncryptionCSPRNG> csprng,
                      bool useSimulation) {

  auto inputTransformers = std::vector<InputTransformer>();

  for (auto gateInfo : info.asReader().getInputs()) {
    InputTransformer transformer;
    if (gateInfo.getTypeInfo().hasIndex()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getIndexInputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasPlaintext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getPlaintextInputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasLweCiphertext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getLweCiphertextInputTransformer(
                      keyset, gateInfo, csprng, useSimulation));
    } else {
      return StringError("Malformed input gate info.");
    }
    inputTransformers.push_back(transformer);
  }

  auto outputTransformers = std::vector<OutputTransformer>();

  for (auto gateInfo : info.asReader().getOutputs()) {
    OutputTransformer transformer;
    if (gateInfo.getTypeInfo().hasIndex()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getIndexOutputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasPlaintext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getPlaintextOutputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasLweCiphertext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getLweCiphertextOutputTransformer(
                      keyset, gateInfo, useSimulation));
    } else {
      return StringError("Malformed output gate info.");
    }
    outputTransformers.push_back(transformer);
  }

  return ClientCircuit(info, inputTransformers, outputTransformers,
                       useSimulation);
}

Result<ClientCircuit> ClientCircuit::createEncrypted(
    const Message<concreteprotocol::CircuitInfo> &info,
    const ClientKeyset &keyset,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng) {
  return ClientCircuit::create(info, keyset, csprng, false);
}

Result<ClientCircuit> ClientCircuit::createSimulated(
    const Message<concreteprotocol::CircuitInfo> &info,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng) {
  return ClientCircuit::create(info, ClientKeyset(), csprng, true);
}

Result<TransportValue> ClientCircuit::prepareInput(Value arg, size_t pos) {
  if (simulated) {
    return StringError("Called prepareInput on simulated client circuit.");
  }
  if (pos >= inputTransformers.size()) {
    return StringError("Tried to prepare a Value for incorrect position.");
  }
  return inputTransformers[pos](arg);
}

Result<Value> ClientCircuit::processOutput(TransportValue result, size_t pos) {
  if (simulated) {
    return StringError("Called processOutput on simulated client circuit.");
  }
  if (pos >= outputTransformers.size()) {
    return StringError(
        "Tried to process a TransportValue for incorrect position.");
  }
  return outputTransformers[pos](result);
}

Result<TransportValue> ClientCircuit::simulatePrepareInput(Value arg,
                                                           size_t pos) {
  if (!simulated) {
    return StringError(
        "Called simulatePrepareInput on encrypted client circuit.");
  }
  if (pos >= inputTransformers.size()) {
    return StringError("Tried to prepare a Value for incorrect position.");
  }
  return inputTransformers[pos](arg);
}

Result<Value> ClientCircuit::simulateProcessOutput(TransportValue result,
                                                   size_t pos) {
  if (!simulated) {
    return StringError(
        "Called simulateProcessOutput on encrypted client circuit.");
  }
  if (pos >= outputTransformers.size()) {
    return StringError(
        "Tried to process a TransportValue for incorrect position.");
  }
  return outputTransformers[pos](result);
}

std::string ClientCircuit::getName() {
  return circuitInfo.asReader().getName();
}

const Message<concreteprotocol::CircuitInfo> &ClientCircuit::getCircuitInfo() {
  return circuitInfo;
}

Result<ClientProgram> ClientProgram::createEncrypted(
    const Message<concreteprotocol::ProgramInfo> &info,
    const ClientKeyset &keyset,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng) {
  ClientProgram output;
  for (auto circuitInfo : info.asReader().getCircuits()) {
    OUTCOME_TRY(const ClientCircuit clientCircuit,
                ClientCircuit::createEncrypted(circuitInfo, keyset, csprng));
    output.circuits.push_back(clientCircuit);
  }
  return output;
}

Result<ClientProgram> ClientProgram::createSimulated(
    const Message<concreteprotocol::ProgramInfo> &info,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng) {
  ClientProgram output;
  for (auto circuitInfo : info.asReader().getCircuits()) {
    OUTCOME_TRY(const ClientCircuit clientCircuit,
                ClientCircuit::createSimulated(circuitInfo, csprng));
    output.circuits.push_back(clientCircuit);
  }
  return output;
}

Result<ClientCircuit> ClientProgram::getClientCircuit(std::string circuitName) {
  for (auto circuit : circuits) {
    if (circuit.getName() == circuitName) {
      return circuit;
    }
  }
  return StringError("Tried to get unknown client circuit: `" + circuitName +
                     "`");
}

Result<TransportValue>
importTfhersFheUint8(llvm::ArrayRef<uint8_t> serializedFheUint8,
                     TfhersFheIntDescription desc, uint32_t encryptionKeyId,
                     double encryptionVariance) {
  if (desc.width != 8 || desc.is_signed == true) {
    return StringError(
        "trying to import FheUint8 but description doesn't match this type");
  }

  auto dims = std::vector({desc.n_cts, desc.lwe_size});
  auto outputTensor = Tensor<uint64_t>::fromDimensions(dims);
  auto err = concrete_cpu_tfhers_uint8_to_lwe_array(
      serializedFheUint8.data(), serializedFheUint8.size(),
      outputTensor.values.data(), desc);
  if (err) {
    return StringError("couldn't convert fheuint to lwe array: err()")
           << err << ")";
  }

  auto value = Value{outputTensor}.intoRawTransportValue();
  auto lwe = value.asBuilder().initTypeInfo().initLweCiphertext();
  lwe.setIntegerPrecision(64);
  // dimensions
  lwe.initAbstractShape().setDimensions({(uint32_t)desc.n_cts});
  lwe.initConcreteShape().setDimensions(
      {(uint32_t)desc.n_cts, (uint32_t)desc.lwe_size});
  // encryption
  auto encryption = lwe.initEncryption();
  encryption.setLweDimension((uint32_t)desc.lwe_size - 1);
  encryption.initModulus().initMod().initNative();
  encryption.setKeyId(encryptionKeyId);
  encryption.setVariance(encryptionVariance);
  // Encoding
  auto encoding = lwe.initEncoding();
  auto integer = encoding.initInteger();
  integer.setIsSigned(false);
  integer.setWidth(std::log2(desc.message_modulus * desc.carry_modulus));
  integer.initMode().initNative();

  return value;
}

Result<std::vector<uint8_t>>
exportTfhersFheUint8(TransportValue value, TfhersFheIntDescription desc) {
  if (desc.width != 8 || desc.is_signed == true) {
    return StringError(
        "trying to export FheUint8 but description doesn't match this type");
  }

  auto fheuint = Value::fromRawTransportValue(value);
  if (fheuint.isScalar()) {
    return StringError("expected a tensor, but value is a scalar");
  }
  auto tensorOrError = fheuint.getTensor<uint64_t>();
  if (!tensorOrError.has_value()) {
    return StringError("couldn't get tensor from value");
  }
  const size_t bufferSize =
      concrete_cpu_tfhers_fheint_buffer_size_u64(desc.lwe_size, desc.n_cts);
  std::vector<uint8_t> buffer(bufferSize, 0);
  auto flatData = tensorOrError.value().values;
  auto size = concrete_cpu_lwe_array_to_tfhers_uint8(
      flatData.data(), buffer.data(), buffer.size(), desc);
  if (size == 0) {
    return StringError("couldn't convert lwe array to fheuint8");
  }
  // we truncate to the serialized data
  assert(size <= buffer.size());
  buffer.resize(size, 0);
  return buffer;
}

Result<TransportValue>
importTfhersFheInt8(llvm::ArrayRef<uint8_t> serializedFheUint8,
                    TfhersFheIntDescription desc, uint32_t encryptionKeyId,
                    double encryptionVariance) {
  if (desc.width != 8 || desc.is_signed == false) {
    return StringError(
        "trying to import FheInt8 but description doesn't match this type");
  }

  auto dims = std::vector({desc.n_cts, desc.lwe_size});
  auto outputTensor = Tensor<uint64_t>::fromDimensions(dims);
  auto err = concrete_cpu_tfhers_int8_to_lwe_array(
      serializedFheUint8.data(), serializedFheUint8.size(),
      outputTensor.values.data(), desc);
  if (err) {
    return StringError("couldn't convert fheint to lwe array");
  }

  auto value = Value{outputTensor}.intoRawTransportValue();
  auto lwe = value.asBuilder().initTypeInfo().initLweCiphertext();
  lwe.setIntegerPrecision(64);
  // dimensions
  lwe.initAbstractShape().setDimensions({(uint32_t)desc.n_cts});
  lwe.initConcreteShape().setDimensions(
      {(uint32_t)desc.n_cts, (uint32_t)desc.lwe_size});
  // encryption
  auto encryption = lwe.initEncryption();
  encryption.setLweDimension((uint32_t)desc.lwe_size - 1);
  encryption.initModulus().initMod().initNative();
  encryption.setKeyId(encryptionKeyId);
  encryption.setVariance(encryptionVariance);
  // Encoding
  auto encoding = lwe.initEncoding();
  auto integer = encoding.initInteger();
  integer.setIsSigned(false);
  integer.setWidth(std::log2(desc.message_modulus * desc.carry_modulus));
  integer.initMode().initNative();

  return value;
}

Result<std::vector<uint8_t>> exportTfhersFheInt8(TransportValue value,
                                                 TfhersFheIntDescription desc) {
  if (desc.width != 8 || desc.is_signed == false) {
    return StringError(
        "trying to export FheInt8 but description doesn't match this type");
  }

  auto fheuint = Value::fromRawTransportValue(value);
  if (fheuint.isScalar()) {
    return StringError("expected a tensor, but value is a scalar");
  }
  auto tensorOrError = fheuint.getTensor<uint64_t>();
  if (!tensorOrError.has_value()) {
    return StringError("couldn't get tensor from value");
  }
  size_t buffer_size =
      concrete_cpu_tfhers_fheint_buffer_size_u64(desc.lwe_size, desc.n_cts);
  std::vector<uint8_t> buffer(buffer_size, 0);
  auto flat_data = tensorOrError.value().values;
  auto size = concrete_cpu_lwe_array_to_tfhers_int8(
      flat_data.data(), buffer.data(), buffer.size(), desc);
  if (size == 0) {
    return StringError("couldn't convert lwe array to fheint8");
  }
  // we truncate to the serialized data
  assert(size <= buffer.size());
  buffer.resize(size, 0);
  return buffer;
}

} // namespace clientlib
} // namespace concretelang

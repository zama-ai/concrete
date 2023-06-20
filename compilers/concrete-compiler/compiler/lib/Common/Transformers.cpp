// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Transformers.h"
#include "concrete-cpu.h"
#include "concrete-protocol.pb.h"
#include "concretelang/Common/CRT.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/simulation.h"
#include <google/protobuf/util/message_differencer.h>
#include <stdlib.h>

using concretelang::error::Result;
using concretelang::keysets::ClientKeyset;
using concretelang::values::Tensor;
using concretelang::values::TransportValue;
using concretelang::values::Value;
using google::protobuf::util::MessageDifferencer;

namespace concretelang {
namespace transformers {

/// A private type for transformers working purely on values.
typedef std::function<Value(Value)> Transformer;

Result<void> checkValueRawProps(Value &val,
                                const concreteprotocol::GateInfo &gateInfo) {
  if (!val.isCompatibleWithShape(gateInfo.rawinfo().shape())) {
    return StringError("Tried to transform value with incompatible shape.");
  }
  if (val.getIntegerPrecision() != gateInfo.rawinfo().integerprecision()) {
    return StringError("Tried to transform value with incompatible integer "
                       "precision.");
  }
  if (val.isSigned() != gateInfo.rawinfo().issigned()) {
    return StringError(
        "Tried to transform value with incompatible integer signedness.");
  }

  return outcome::success();
}

Result<void> checkValueIndexProps(Value &val,
                                  const concreteprotocol::GateInfo &gateInfo) {
  if (!val.isCompatibleWithShape(gateInfo.index().shape())) {
    return StringError(
        "Tried to transform index value with incompatible shape.");
  }
  if (val.getIntegerPrecision() != gateInfo.index().integerprecision()) {
    return StringError(
        "Tried to transform index value with incompatible integer "
        "precision.");
  }
  if (val.isSigned() != gateInfo.index().issigned()) {
    return StringError(
        "Tried to transform index value with incompatible integer "
        "signedness.");
  }

  return outcome::success();
}

Result<void>
checkValuePlaintextProps(Value &val,
                         const concreteprotocol::GateInfo &gateInfo) {
  if (!val.isCompatibleWithShape(gateInfo.index().shape())) {
    return StringError(
        "Tried to transform plaintext value with incompatible shape.");
  }
  if (val.getIntegerPrecision() != gateInfo.index().integerprecision()) {
    return StringError(
        "Tried to transform plaintext value with incompatible integer "
        "precision.");
  }
  if (val.isSigned() != gateInfo.index().issigned()) {
    return StringError("Tried to transform plaintext value with incompatible "
                       "integer signedness.");
  }

  return outcome::success();
}

Result<void>
checkValueLweCiphertextProps(Value &val,
                             const concreteprotocol::GateInfo &gateInfo) {
  if (!val.isCompatibleWithShape(gateInfo.lweciphertext().concreteshape())) {
    return StringError("Tried to transform ciphertext value with "
                       "incompatible shape.");
  }
  if (val.getIntegerPrecision() !=
      gateInfo.lweciphertext().integerprecision()) {
    return StringError(
        "Tried to transform ciphertext value with incompatible integer "
        "precision.");
  }
  if (val.isSigned()) {
    return StringError(
        "Tried to transform ciphertext value with incompatible signedness.");
  }

  return outcome::success();
}

Result<void>
checkTransportValueRawProps(TransportValue &transportVal,
                            const concreteprotocol::GateInfo &gateInfo) {
  if (!MessageDifferencer::Equals(gateInfo.rawinfo().shape(),
                                  transportVal.rawinfo().shape())) {
    return StringError(
        "Tried to transform transport value with incompatible shape.");
  }
  if (transportVal.rawinfo().integerprecision() !=
      gateInfo.rawinfo().integerprecision()) {
    return StringError(
        "Tried to transform transport value with incompatible integer "
        "precision.");
  }
  if (transportVal.rawinfo().issigned() != gateInfo.rawinfo().issigned()) {
    return StringError(
        "Tried to transform transport value with incompatible integer "
        "signedness.");
  }

  return outcome::success();
}

Result<void>
checkTransportValueIndexProps(TransportValue &transportVal,
                              const concreteprotocol::GateInfo &gateInfo) {
  if (!transportVal.valueinfo().has_index()) {
    return StringError(
        "Tried to transform transport value with incompatible type.");
  }
  if (!google::protobuf::util::MessageDifferencer::Equals(
          gateInfo.index().shape(), transportVal.valueinfo().index().shape())) {
    return StringError(
        "Tried to transform index transport value with incompatible shape.");
  }
  if (transportVal.valueinfo().index().integerprecision() !=
      gateInfo.index().integerprecision()) {
    return StringError(
        "Tried to transform index transport value with incompatible integer "
        "precision.");
  }
  if (transportVal.valueinfo().index().issigned() !=
      gateInfo.index().issigned()) {
    return StringError(
        "Tried to transform index transport value with incompatible integer "
        "signedness.");
  }

  return outcome::success();
}

Result<void>
checkTransportValuePlaintextProps(TransportValue &transportVal,
                                  const concreteprotocol::GateInfo &gateInfo) {
  if (!transportVal.valueinfo().has_plaintext()) {
    return StringError(
        "Tried to transform transport value with incompatible type.");
  }
  if (!google::protobuf::util::MessageDifferencer::Equals(
          gateInfo.plaintext().shape(),
          transportVal.valueinfo().plaintext().shape())) {
    return StringError("Tried to transform plaintext transport value with "
                       "incompatible shape.");
  }
  if (transportVal.valueinfo().plaintext().integerprecision() !=
      gateInfo.plaintext().integerprecision()) {
    return StringError("Tried to transform plaintext transport value with "
                       "incompatible integer precision.");
  }
  if (transportVal.valueinfo().plaintext().issigned() !=
      gateInfo.plaintext().issigned()) {
    return StringError("Tried to transform plaintext transport value with "
                       "incompatible integer signedness.");
  }

  return outcome::success();
}

Result<void> checkTransportValueLweCiphertextProps(
    TransportValue &transportVal, const concreteprotocol::GateInfo &gateInfo) {
  if (!transportVal.valueinfo().has_lweciphertext()) {
    return StringError(
        "Tried to transform transport value with incompatible type.");
  }
  if (!google::protobuf::util::MessageDifferencer::Equals(
          gateInfo.lweciphertext().concreteshape(),
          transportVal.valueinfo().lweciphertext().concreteshape())) {
    return StringError("Tried to transform ciphertext transport value with "
                       "incompatible shape.");
  }
  if (transportVal.valueinfo().lweciphertext().integerprecision() !=
      gateInfo.lweciphertext().integerprecision()) {
    return StringError("Tried to transform ciphertext transport value with "
                       "incompatible integer precision.");
  }
  if (transportVal.valueinfo().lweciphertext().lwedimension() !=
      gateInfo.lweciphertext().encryption().lwedimension()) {
    return StringError("Tried to transform ciphertext transport value with "
                       "incompatible lwe dimension.");
  }
  if (!google::protobuf::util::MessageDifferencer::Equals(
          gateInfo.lweciphertext().encryption().modulus(),
          transportVal.valueinfo().lweciphertext().modulus())) {
    return StringError("Tried to transform ciphertext transport value with "
                       "incompatible modulus.");
  }
  if (transportVal.valueinfo().lweciphertext().compression() !=
      gateInfo.lweciphertext().compression()) {
    return StringError("Tried to transform ciphertext transport value with "
                       "incompatible lwe compression.");
  }

  return outcome::success();
}

Result<Transformer> getBooleanEncodingTransformer() {
  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      outputTensor.values[i] = inputTensor.values[i] << 62;
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getNativeModeIntegerEncodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  auto width = info.width();

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      outputTensor.values[i] = inputTensor.values[i] << (64 - (width + 1));
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getNativeModeIntegerDecodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  auto precision = info.width();
  auto isSigned = info.issigned();

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      auto input = inputTensor.values[i];

      // Decode unsigned integer
      uint64_t output = input >> (64 - precision - 2);
      auto carry = output % 2;
      uint64_t mod = (((uint64_t)1) << (precision + 1));
      output = ((output >> 1) + carry) % mod;

      // Further decode signed integers.
      if (isSigned) {
        uint64_t maxPos = (((uint64_t)1) << (precision - 1));
        if (output >= maxPos) { // The output is actually negative.
          // Set the preceding bits to zero
          output |= UINT64_MAX << precision;
          // This makes sure when the value is cast to int64, it has the
          // correct value
        };
      }

      outputTensor.values[i] = output;
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getChunkedModeIntegerEncodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  auto size = info.chunked().size();
  auto chunkWidth = info.chunked().width();
  uint64_t mask = (1 << chunkWidth) - 1;

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.push_back(size);
    outputTensor.values.resize(outputTensor.values.size() * size);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      auto value = inputTensor.values[i];
      for (size_t j = 0; j < size; j++) {
        auto chunk = value & mask;
        outputTensor.values[i * size + j] = ((uint64_t)chunk)
                                            << (64 - (chunkWidth + 1));
        value >>= chunkWidth;
      }
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getChunkedModeIntegerDecodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  auto chunkSize = info.chunked().size();
  auto chunkWidth = info.chunked().width();
  auto isSigned = info.issigned();
  uint64_t mask = (1 << chunkWidth) - 1;

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.pop_back();
    outputTensor.values.resize(outputTensor.values.size() / chunkSize);

    for (size_t i = 0; i < outputTensor.values.size(); i++) {
      uint64_t output = 0;
      for (size_t j = 0; j < chunkSize; j++) {
        auto input = inputTensor.values[i * chunkSize + j];

        // Decode unsigned integer
        uint64_t chunkOutput = input >> (64 - chunkWidth - 2);
        auto carry = chunkOutput % 2;
        uint64_t mod = (((uint64_t)1) << (chunkWidth + 1));
        chunkOutput = ((chunkOutput >> 1) + carry) % mod;

        // Further decode signed integers.
        if (isSigned) {
          uint64_t maxPos = (((uint64_t)1) << (chunkWidth - 1));
          if (output >= maxPos) { // The output is actually negative.
            // Set the preceding bits to zero
            chunkOutput |= UINT64_MAX << chunkWidth;
            // This makes sure when the value is cast to int64, it has the
            // correct value
          };
        }

        chunkOutput &= mask;
        output += chunkOutput << (chunkWidth * j);
      }
      outputTensor.values[i] = output;
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getCrtModeIntegerEncodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  std::vector<int64_t> moduli;
  for (auto modulus : info.crt().moduli()) {
    moduli.push_back(modulus);
  }
  auto size = info.crt().moduli_size();
  auto productOfModuli = concretelang::crt::productOfModuli(moduli);

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.push_back(size);
    outputTensor.values.resize(outputTensor.values.size() * size);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      auto value = inputTensor.values[i];
      for (size_t j = 0; j < (size_t)size; j++) {
        outputTensor.values[i * size + j] =
            concretelang::crt::encode(value, moduli[j], productOfModuli);
      }
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getCrtModeIntegerDecodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  std::vector<int64_t> moduli;
  for (auto modulus : info.crt().moduli()) {
    moduli.push_back(modulus);
  }
  std::vector<int64_t> remainders(info.crt().moduli_size());
  auto size = info.crt().moduli_size();
  auto isSigned = info.issigned();

  return [=](Value input) mutable {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.pop_back();
    outputTensor.values.resize(outputTensor.values.size() / size);

    for (size_t i = 0; i < outputTensor.values.size(); i++) {
      for (size_t j = 0; j < (size_t)size; j++) {
        remainders[j] =
            crt::decode(inputTensor.values[i * size + j], moduli[j]);
      }

      // Compute the inverse crt
      uint64_t output = crt::iCrt(moduli, remainders);

      // Further decode signed integers
      if (isSigned) {
        uint64_t maxPos = 1;
        for (auto prime : moduli) {
          maxPos *= prime;
        }
        maxPos /= 2;
        if (output >= maxPos) {
          output -= maxPos * 2;
        }
      }
      outputTensor.values[i] = output;
    }

    return Value{outputTensor};
  };
}

Result<Transformer>
getEncryptionTransformer(ClientKeyset keyset,
                         concreteprotocol::LweCiphertextEncryptionInfo info,
                         CSPRNG &csprng) {

  auto key = keyset.lweSecretKeys[info.keyid()];
  auto lweDimension = info.lwedimension();
  auto lweSize = lweDimension + 1;
  auto variance = info.variance();

  return [=, &csprng](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.push_back(lweSize);
    outputTensor.values.resize(outputTensor.values.size() * lweSize);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      concrete_cpu_encrypt_lwe_ciphertext_u64(
          key.getRawPtr(), &outputTensor.values[i * lweSize],
          inputTensor.values[i], lweDimension, variance, csprng.ptr,
          csprng.vtable);
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getEncryptionSimulationTransformer(
    concreteprotocol::LweCiphertextEncryptionInfo info,
    CSPRNG &csprng) {

  auto lweDimension = info.lwedimension();

  return [=, &csprng](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      outputTensor.values[i] = sim_encrypt_lwe_u64(
          inputTensor.values[i], lweDimension, (void *)csprng.ptr);
    }

    return Value{outputTensor};
  };
}

Result<Transformer>
getDecryptionTransformer(ClientKeyset keyset,
                         concreteprotocol::LweCiphertextEncryptionInfo info) {

  auto key = keyset.lweSecretKeys[info.keyid()];
  auto lweDimension = info.lwedimension();
  auto lweSize = lweDimension + 1;

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.pop_back();
    outputTensor.values.resize(outputTensor.values.size() / lweSize);

    for (size_t i = 0; i < outputTensor.values.size(); i++) {
      concrete_cpu_decrypt_lwe_ciphertext_u64(
          key.getRawPtr(), &inputTensor.values[i * lweSize], lweDimension,
          &outputTensor.values[i]);
    }

    return Value{outputTensor};
  };
}

Result<Transformer>
getDecryptionSimulationTransformer() {
  return [](auto input) { return input; };
}

Result<Transformer> getNoneCompressionTransformer() {
  return [](auto input) { return input; };
}

Result<Transformer> getNoneDecompressionTransformer() {
  return [](auto input) { return input; };
}

Result<Transformer> getBooleanDecodingTransformer() {
  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      auto input = inputTensor.values[i];

      uint64_t output = input >> 60;
      auto carry = output % 2;
      uint64_t mod = (((uint64_t)1) << 3);
      output = ((output >> 1) + carry) % mod;

      outputTensor.values[i] = output;
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getIntegerEncodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  if (info.has_native()) {
    return getNativeModeIntegerEncodingTransformer(info);
  } else if (info.has_chunked()) {
    return getChunkedModeIntegerEncodingTransformer(info);
  } else if (info.has_crt()) {
    return getCrtModeIntegerEncodingTransformer(info);
  } else {
    return StringError(
        "Tried to construct integer encoding transformer without mode.");
  }
}

Result<Transformer> getIntegerDecodingTransformer(
    const concreteprotocol::IntegerCiphertextEncodingInfo &info) {
  if (info.has_native()) {
    return getNativeModeIntegerDecodingTransformer(info);
  } else if (info.has_chunked()) {
    return getChunkedModeIntegerDecodingTransformer(info);
  } else if (info.has_crt()) {
    return getCrtModeIntegerDecodingTransformer(info);
  } else {
    return StringError(
        "Tried to construct integer decoding transformer without mode.");
  }
}

Result<InputTransformer> TransformerFactory::getIndexInputTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_index()) {
    return StringError(
        "Tried to get index input transformer from non-index gate info.");
  }
  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(checkValueRawProps(val, gateInfo));
    OUTCOME_TRYV(checkValueIndexProps(val, gateInfo));
    auto output = val.intoRawTransportValue();
    auto indexValueInfo = new concreteprotocol::IndexValueInfo();
    indexValueInfo->set_allocated_shape(
        new concreteprotocol::Shape(output.rawinfo().shape()));
    indexValueInfo->set_integerprecision(output.rawinfo().integerprecision());
    indexValueInfo->set_issigned(output.rawinfo().issigned());
    output.mutable_valueinfo()->set_allocated_index(indexValueInfo);
    return output;
  };
}

Result<OutputTransformer> TransformerFactory::getIndexOutputTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_index()) {
    return StringError(
        "Tried to get index output transformer from non-index gate info.");
  }
  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(checkTransportValueRawProps(transportVal, gateInfo));
    OUTCOME_TRYV(checkTransportValueIndexProps(transportVal, gateInfo));
    return Value::fromRawTransportValue(transportVal);
  };
}

Result<ArgTransformer> TransformerFactory::getIndexArgTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_index()) {
    return StringError(
        "Tried to get index arg transformer from non-index gate info.");
  }
  return getIndexOutputTransformer(gateInfo);
}

Result<ReturnTransformer> TransformerFactory::getIndexReturnTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_index()) {
    return StringError(
        "Tried to get index return transformer from non-index gate info.");
  }
  return getIndexInputTransformer(gateInfo);
}

Result<InputTransformer> TransformerFactory::getPlaintextInputTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_plaintext()) {
    return StringError("Tried to get plaintext input transformer from "
                       "non-plaintext gate info.");
  }
  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(checkValueRawProps(val, gateInfo));
    OUTCOME_TRYV(checkValuePlaintextProps(val, gateInfo));
    auto output = val.intoRawTransportValue();
    auto plaintextValueInfo = new concreteprotocol::PlaintextValueInfo();
    plaintextValueInfo->set_allocated_shape(
        new concreteprotocol::Shape(output.rawinfo().shape()));
    plaintextValueInfo->set_integerprecision(
        output.rawinfo().integerprecision());
    plaintextValueInfo->set_issigned(output.rawinfo().issigned());
    output.mutable_valueinfo()->set_allocated_plaintext(plaintextValueInfo);
    return output;
  };
}

Result<OutputTransformer> TransformerFactory::getPlaintextOutputTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_plaintext()) {
    return StringError("Tried to get plaintext output transformer from "
                       "non-plaintext gate info.");
  }
  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(checkTransportValueRawProps(transportVal, gateInfo));
    OUTCOME_TRYV(checkTransportValuePlaintextProps(transportVal, gateInfo));
    return Value::fromRawTransportValue(transportVal);
  };
}

Result<ArgTransformer> TransformerFactory::getPlaintextArgTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_plaintext()) {
    return StringError("Tried to get plaintext arg transformer from "
                       "non-plaintext gate info.");
  }
  return getPlaintextOutputTransformer(gateInfo);
}

Result<ReturnTransformer> TransformerFactory::getPlaintextReturnTransformer(
    concreteprotocol::GateInfo gateInfo) {
  if (!gateInfo.has_plaintext()) {
    return StringError("Tried to get plaintext return transformer from "
                       "non-plaintext gate info.");
  }
  return getPlaintextInputTransformer(gateInfo);
}

Result<InputTransformer> TransformerFactory::getLweCiphertextInputTransformer(
    ClientKeyset keyset, concreteprotocol::GateInfo gateInfo, CSPRNG &csprng,
    bool useSimulation) {
  if (!gateInfo.has_lweciphertext()) {
    return StringError("Tried to get lwe ciphertext input transformer from "
                       "non-ciphertext gate info.");
  }
  auto keyid = gateInfo.lweciphertext().encryption().keyid();
  if (keyset.lweSecretKeys.size() >= keyid) {
    return StringError(
        "Tried to generate lwe ciphertext input transformer with "
        "key id unavailable");
  }

  /// Generating the encoding transformer.
  Transformer encodingTransformer;
  if (gateInfo.lweciphertext().has_boolean()) {
    OUTCOME_TRY(encodingTransformer, getBooleanEncodingTransformer());
  } else if (gateInfo.lweciphertext().has_integer()) {
    OUTCOME_TRY(encodingTransformer, getIntegerEncodingTransformer(
                                         gateInfo.lweciphertext().integer()));
  } else {
    return StringError("Malformed gate info");
  }

  /// Generating the encryption transformer.
  Transformer encryptionTransformer;
  if (useSimulation) {
    OUTCOME_TRY(encryptionTransformer,
                getEncryptionSimulationTransformer(
                    gateInfo.lweciphertext().encryption(), csprng));
  } else {
    OUTCOME_TRY(encryptionTransformer,
                getEncryptionTransformer(
                    keyset, gateInfo.lweciphertext().encryption(), csprng));
  }

  /// Generating the compression transformer.
  Transformer compressionTransformer;
  if (gateInfo.lweciphertext().compression() ==
      concreteprotocol::Compression::none) {
    OUTCOME_TRY(compressionTransformer, getNoneCompressionTransformer());
  } else {
    return StringError(
        "Only none compression is currently supported for lwe ciphertext "
        "currently.");
  }

  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(checkValueRawProps(val, gateInfo));
    if (!useSimulation){
      OUTCOME_TRYV(checkValueLweCiphertextProps(val, gateInfo));
    }
    auto output =
        compressionTransformer(encryptionTransformer(encodingTransformer(val)))
            .intoRawTransportValue();
    auto lweCiphertextValueInfo =
        new concreteprotocol::LweCiphertextValueInfo();
    lweCiphertextValueInfo->set_allocated_concreteshape(
        new concreteprotocol::Shape(gateInfo.lweciphertext().concreteshape()));
    lweCiphertextValueInfo->set_integerprecision(
        gateInfo.lweciphertext().integerprecision());
    lweCiphertextValueInfo->set_lwedimension(
        gateInfo.lweciphertext().encryption().lwedimension());
    lweCiphertextValueInfo->set_allocated_modulus(new concreteprotocol::Modulus(
        gateInfo.lweciphertext().encryption().modulus()));
    lweCiphertextValueInfo->set_compression(
        gateInfo.lweciphertext().compression());
    output.mutable_valueinfo()->set_allocated_lweciphertext(
        lweCiphertextValueInfo);
    return output;
  };
}

Result<OutputTransformer> TransformerFactory::getLweCiphertextOutputTransformer(
    ClientKeyset keyset, concreteprotocol::GateInfo gateInfo,
    bool useSimulation) {
  if (!gateInfo.has_lweciphertext()) {
    return StringError("Tried to get lwe ciphertext output transformer from "
                       "non-ciphertext gate info.");
  }
  auto keyid = gateInfo.lweciphertext().encryption().keyid();
  if (keyset.lweSecretKeys.size() >= keyid) {
    return StringError(
        "Tried to generate lwe ciphertext output transformer with "
        "key id unavailable");
  }

  /// Generating the decompression transformer.
  Transformer decompressionTransformer;
  if (gateInfo.lweciphertext().compression() ==
      concreteprotocol::Compression::none) {
    OUTCOME_TRY(decompressionTransformer, getNoneDecompressionTransformer());
  } else {
    return StringError(
        "Only none compression is currently supported for lwe ciphertext "
        "currently.");
  }

  /// Generating the decryption transformer.
  Transformer decryptionTransformer;
  if (useSimulation) {
    OUTCOME_TRY(decryptionTransformer, getDecryptionSimulationTransformer());
  } else {
    OUTCOME_TRY(decryptionTransformer,
                getDecryptionTransformer(
                    keyset, gateInfo.lweciphertext().encryption()));
  }

  /// Generating the decoding transformer.
  Transformer decodingTransformer;
  if (gateInfo.lweciphertext().has_boolean()) {
    OUTCOME_TRY(decodingTransformer, getBooleanDecodingTransformer());
  } else if (gateInfo.lweciphertext().has_integer()) {
    OUTCOME_TRY(decodingTransformer, getIntegerDecodingTransformer(
                                         gateInfo.lweciphertext().integer()));
  } else {
    return StringError("Malformed gate info");
  }

  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(checkTransportValueRawProps(transportVal, gateInfo));
    if (!useSimulation){
      OUTCOME_TRYV(checkTransportValueLweCiphertextProps(transportVal, gateInfo));
    }
    return decodingTransformer(decryptionTransformer(
        decompressionTransformer(Value::fromRawTransportValue(transportVal))));
  };
}

Result<ArgTransformer> TransformerFactory::getLweCiphertextArgTransformer(
    concreteprotocol::GateInfo gateInfo, bool useSimulation) {
  if (!gateInfo.has_lweciphertext()) {
    return StringError("Tried to get lwe ciphertext arg transformer from "
                       "non-ciphertext gate info.");
  }

  /// Generating the decompression transformer.
  Transformer decompressionTransformer;
  if (gateInfo.lweciphertext().compression() ==
      concreteprotocol::Compression::none) {
    OUTCOME_TRY(decompressionTransformer, getNoneDecompressionTransformer());
  } else {
    return StringError(
        "Only none compression is currently supported for lwe ciphertext "
        "currently.");
  }

  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(checkTransportValueRawProps(transportVal, gateInfo));
    if (!useSimulation){
      OUTCOME_TRYV(checkTransportValueLweCiphertextProps(transportVal, gateInfo));
    }
    return decompressionTransformer(Value::fromRawTransportValue(transportVal));
  };
}

Result<ReturnTransformer> TransformerFactory::getLweCiphertextReturnTransformer(
    concreteprotocol::GateInfo gateInfo, bool useSimulation) {
  if (!gateInfo.has_lweciphertext()) {
    return StringError("Tried to get lwe ciphertext return transformer from "
                       "non-ciphertext gate info.");
  }

  /// Generating the compression transformer.
  Transformer compressionTransformer;
  if (gateInfo.lweciphertext().compression() ==
      concreteprotocol::Compression::none) {
    OUTCOME_TRY(compressionTransformer, getNoneCompressionTransformer());
  } else {
    return StringError(
        "Only none compression is currently supported for lwe ciphertext "
        "currently.");
  }

  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(checkValueRawProps(val, gateInfo));
    if (!useSimulation){
      OUTCOME_TRYV(checkValueLweCiphertextProps(val, gateInfo));
    }
    auto output = compressionTransformer(val).intoRawTransportValue();
    auto lweCiphertextValueInfo =
        new concreteprotocol::LweCiphertextValueInfo();
    lweCiphertextValueInfo->set_allocated_concreteshape(
        new concreteprotocol::Shape(gateInfo.lweciphertext().concreteshape()));
    lweCiphertextValueInfo->set_integerprecision(
        gateInfo.lweciphertext().integerprecision());
    lweCiphertextValueInfo->set_lwedimension(
        gateInfo.lweciphertext().encryption().lwedimension());
    lweCiphertextValueInfo->set_allocated_modulus(new concreteprotocol::Modulus(
        gateInfo.lweciphertext().encryption().modulus()));
    lweCiphertextValueInfo->set_compression(
        gateInfo.lweciphertext().compression());
    output.mutable_valueinfo()->set_allocated_lweciphertext(
        lweCiphertextValueInfo);
    return output;
  };
}

} // namespace transformers
} // namespace concretelang

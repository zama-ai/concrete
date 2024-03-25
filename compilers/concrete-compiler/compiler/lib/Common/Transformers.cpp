// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Transformers.h"
#include "capnp/any.h"
#include "concrete-cpu.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/CRT.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Runtime/simulation.h"
#include <memory>
#include <stdlib.h>
#include <string>

using concretelang::error::Result;
using concretelang::keysets::ClientKeyset;
using concretelang::values::getCorrespondingPrecision;
using concretelang::values::Tensor;
using concretelang::values::TransportValue;
using concretelang::values::Value;

namespace concretelang {
namespace transformers {

/// A private type for value verifiers.
typedef std::function<Result<void>(const Value &)> ValueVerifier;

/// A private type for transport value verifiers.
typedef std::function<Result<void>(const TransportValue &)>
    TransportValueVerifier;

/// A private type for transformers working purely on values.
typedef std::function<Value(Value)> Transformer;

Result<ValueVerifier> getIndexInputValueVerifier(
    const Message<concreteprotocol::GateInfo> &gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasIndex()) {
    return StringError("Tried to get index input value verifier for gate info "
                       "without proper type info.");
  }
  return [=](const Value &val) -> Result<void> {
    auto type = gateInfo.asReader().getTypeInfo().getIndex();
    if (!val.isCompatibleWithShape(type.getShape())) {
      return StringError(
          "Tried to transform index value with incompatible shape.");
    }
    if (val.getIntegerPrecision() != type.getIntegerPrecision()) {
      return StringError(
          "Tried to transform index value with incompatible integer "
          "precision.");
    }
    return outcome::success();
  };
}

Result<ValueVerifier> getObliviousValueVerifier() {
  return [=](const Value &val) -> Result<void> { return outcome::success(); };
}

Result<ValueVerifier> getPlaintextInputValueVerifier(
    const Message<concreteprotocol::GateInfo> &gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return StringError("Tried to get plaintext input value verifier for gate "
                       "info without proper type info.");
  }
  return [=](const Value &val) -> Result<void> {
    auto type = gateInfo.asReader().getTypeInfo().getPlaintext();
    if (!val.isCompatibleWithShape(type.getShape())) {
      return StringError(
          "Tried to transform plaintext value with incompatible shape.");
    }
    if (val.getIntegerPrecision() != type.getIntegerPrecision()) {
      return StringError(
          "Tried to transform plaintext value with incompatible integer "
          "precision. Got " +
          std::to_string(val.getIntegerPrecision()) + " expected " +
          std::to_string(gateInfo.asReader()
                             .getTypeInfo()
                             .getPlaintext()
                             .getIntegerPrecision()));
    }
    return outcome::success();
  };
}

Result<ValueVerifier> getLweCiphertextInputValueVerifier(
    const Message<concreteprotocol::GateInfo> &gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return StringError("Tried to get ciphertext input value verifier for gate "
                       "info without proper type info.");
  }

  if (gateInfo.asReader()
          .getTypeInfo()
          .getLweCiphertext()
          .getEncoding()
          .hasBoolean()) {
    return [=](const Value &val) -> Result<void> {
      auto type = gateInfo.asReader().getTypeInfo().getLweCiphertext();
      if (!val.isCompatibleWithShape(type.getAbstractShape())) {
        return StringError("Tried to transform ciphertext input value with "
                           "incompatible shape.");
      }
      if (val.getIntegerPrecision() != 64) {
        return StringError("Tried to transform ciphertext input value "
                           "(boolean) with incompatible integer "
                           "precision. Got " +
                           std::to_string(val.getIntegerPrecision()) +
                           " expected 64");
      }
      if (val.isSigned()) {
        return StringError("Tried to transform ciphertext input value "
                           "(boolean) with incompatible signedness.");
      }
      return outcome::success();
    };
  }

  if (gateInfo.asReader()
          .getTypeInfo()
          .getLweCiphertext()
          .getEncoding()
          .hasInteger()) {
    return [=](const Value &val) -> Result<void> {
      auto type = gateInfo.asReader().getTypeInfo().getLweCiphertext();
      if (!val.isCompatibleWithShape(type.getAbstractShape())) {
        return StringError("Tried to transform ciphertext input value with "
                           "incompatible shape.");
      }
      if (val.getIntegerPrecision() != 64) {
        return StringError("Tried to transform ciphertext input value with "
                           "incompatible integer "
                           "precision. Got " +
                           std::to_string(val.getIntegerPrecision()) +
                           " expected 64.");
      }
      if (val.isSigned() != type.getEncoding().getInteger().getIsSigned()) {
        return StringError("Tried to transform ciphertext input value with "
                           "incompatible signedness.");
      }
      return outcome::success();
    };
  }

  return StringError(
      "Tried to get lwe ciphertext input verifier for wrongly defined gate.");
}

Result<ValueVerifier> getLweCiphertextOutputValueVerifier(
    const Message<concreteprotocol::GateInfo> &gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return StringError("Tried to get ciphertext output value verifier for gate "
                       "info without proper type info.");
  }

  return [=](const Value &val) -> Result<void> {
    auto type = gateInfo.asReader().getTypeInfo().getLweCiphertext();
    if (!val.isCompatibleWithShape(type.getConcreteShape())) {
      return StringError("Tried to transform ciphertext output value with "
                         "incompatible shape.");
    }
    if (val.getIntegerPrecision() != 64) {
      return StringError("Tried to transform ciphertext output value with "
                         "incompatible integer "
                         "precision. Got " +
                         std::to_string(val.getIntegerPrecision()) +
                         " expected 64");
    }
    if (val.isSigned()) {
      return StringError("Tried to transform ciphertext output value with "
                         "incompatible signedness (signed).");
    }
    return outcome::success();
  };
}

Result<TransportValueVerifier> getObliviousTransportValueVerifier() {
  return [=](const TransportValue &val) -> Result<void> {
    return outcome::success();
  };
}

Result<TransportValueVerifier>
getTransportValueVerifier(const Message<concreteprotocol::GateInfo> &gateInfo) {
  return [=](const TransportValue &transportVal) -> Result<void> {
    if (!transportVal.asReader().hasPayload()) {
      return StringError(
          "Tried to transform a transport value without payload.");
    }
    if (!transportVal.asReader().hasRawInfo()) {
      return StringError(
          "Tried to transform a transport value without raw infos.");
    }
    if (!((capnp::AnyStruct::Reader)gateInfo.asReader().getRawInfo() ==
          (capnp::AnyStruct::Reader)transportVal.asReader().getRawInfo())) {
      std::string expected =
          gateInfo.asReader().getRawInfo().toString().flatten().cStr();
      std::string actual =
          transportVal.asReader().getRawInfo().toString().flatten().cStr();
      return StringError("Tried to transform transport value with incompatible "
                         "raw info.\nExpected: " +
                         expected + "\nActual: " + actual);
    }
    size_t expectedPayloadSize =
        transportVal.asReader().getRawInfo().getIntegerPrecision() / 8;
    for (auto dim :
         transportVal.asReader().getRawInfo().getShape().getDimensions()) {
      expectedPayloadSize *= dim;
    }
    size_t actualPayloadSize = 0;
    for (auto blob : transportVal.asReader().getPayload().getData()) {
      actualPayloadSize += blob.size();
    }
    if (actualPayloadSize != expectedPayloadSize) {
      return StringError("Tried to transform a transport value with "
                         "incompatible payload size.");
    }
    if (!transportVal.asReader().getTypeInfo().hasIndex() &&
        !transportVal.asReader().getTypeInfo().hasPlaintext() &&
        !transportVal.asReader().getTypeInfo().hasLweCiphertext()) {
      return StringError(
          "Tried to transform a transport value without type infos.");
    }
    if ((capnp::AnyStruct::Reader)gateInfo.asReader().getTypeInfo() !=
        (capnp::AnyStruct::Reader)transportVal.asReader().getTypeInfo()) {
      std::string expected =
          gateInfo.asReader().getTypeInfo().toString().flatten().cStr();
      std::string actual =
          transportVal.asReader().getTypeInfo().toString().flatten().cStr();
      return StringError("Tried to transform transport value with incompatible "
                         "type info.\nExpected: " +
                         expected + "\nActual: " + actual);
    }
    return outcome::success();
  };
}

Result<Transformer> getBooleanEncodingTransformer() {
  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      outputTensor.values[i] = inputTensor.values[i] << 61;
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getNativeModeIntegerEncodingTransformer(
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  auto width = info.asReader().getWidth();
  auto isSigned = info.asReader().getIsSigned();

  return [=](Value input) {
    Tensor<uint64_t> inputTensor;
    if (isSigned) {
      inputTensor = (Tensor<uint64_t>)input.getTensor<int64_t>().value();
    } else {
      inputTensor = input.getTensor<uint64_t>().value();
    }
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      outputTensor.values[i] = inputTensor.values[i] << (64 - (width + 1));
    }
    return Value{outputTensor};
  };
}

Result<Transformer> getNativeModeIntegerDecodingTransformer(
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  auto precision = info.asReader().getWidth();
  auto isSigned = info.asReader().getIsSigned();

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

    Value output;
    if (isSigned) {
      auto signedOutputTensor = (Tensor<int64_t>)outputTensor;
      output = Value{signedOutputTensor};
    } else {
      output = Value{outputTensor};
    }

    return output;
  };
}

Result<Transformer> getChunkedModeIntegerEncodingTransformer(
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  auto size = info.asReader().getMode().getChunked().getSize();
  auto chunkWidth = info.asReader().getMode().getChunked().getWidth();
  auto isSigned = info.asReader().getIsSigned();
  uint64_t mask = (1 << chunkWidth) - 1;

  return [=](Value input) {
    Tensor<uint64_t> inputTensor;
    if (isSigned) {
      inputTensor = (Tensor<uint64_t>)input.getTensor<int64_t>().value();
    } else {
      inputTensor = input.getTensor<uint64_t>().value();
    }
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
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  auto chunkSize = info.asReader().getMode().getChunked().getSize();
  auto chunkWidth = info.asReader().getMode().getChunked().getWidth();
  auto isSigned = info.asReader().getIsSigned();
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

    Value output;
    if (isSigned) {
      auto signedOutputTensor = (Tensor<int64_t>)outputTensor;
      output = Value{signedOutputTensor};
    } else {
      output = Value{outputTensor};
    }

    return output;
  };
}

Result<Transformer> getCrtModeIntegerEncodingTransformer(
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  std::vector<int64_t> moduli;
  for (auto modulus : info.asReader().getMode().getCrt().getModuli()) {
    moduli.push_back(modulus);
  }
  auto size = info.asReader().getMode().getCrt().getModuli().size();
  auto productOfModuli = concretelang::crt::productOfModuli(moduli);
  auto isSigned = info.asReader().getIsSigned();

  return [=](Value input) {
    Tensor<uint64_t> inputTensor;
    if (isSigned) {
      inputTensor = (Tensor<uint64_t>)input.getTensor<int64_t>().value();
    } else {
      inputTensor = input.getTensor<uint64_t>().value();
    }
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
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> info) {
  std::vector<int64_t> moduli;
  for (auto modulus : info.asReader().getMode().getCrt().getModuli()) {
    moduli.push_back(modulus);
  }
  std::vector<int64_t> remainders(
      info.asReader().getMode().getCrt().getModuli().size());
  auto size = info.asReader().getMode().getCrt().getModuli().size();
  auto isSigned = info.asReader().getIsSigned();

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

    Value output;
    if (isSigned) {
      auto signedOutputTensor = (Tensor<int64_t>)outputTensor;
      output = Value{signedOutputTensor};
    } else {
      output = Value{outputTensor};
    }

    return output;
  };
}

Result<Transformer> getEncryptionTransformer(
    keysets::ClientPublicKeyset keyset,
    const Message<concreteprotocol::LweCiphertextEncryptionInfo> &info,
    std::shared_ptr<csprng::SecretCSPRNG> csprng) {
  OUTCOME_TRY(auto key, keyset.getLwePublicKey(info.asReader().getKeyId()));
  auto lweDimension = info.asReader().getLweDimension();
  auto lweSize = lweDimension + 1;

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.push_back(lweSize);
    outputTensor.values.resize(outputTensor.values.size() * lweSize);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      key.encrypt(&outputTensor.values[i * lweSize], inputTensor.values[i],
                  *csprng);
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getEncryptionTransformer(
    ClientKeyset keyset,
    const Message<concreteprotocol::LweCiphertextEncryptionInfo> &info,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng) {

  auto key = keyset.lweSecretKeys[info.asReader().getKeyId()];
  auto lweDimension = info.asReader().getLweDimension();
  auto lweSize = lweDimension + 1;
  auto variance = info.asReader().getVariance();

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.push_back(lweSize);
    outputTensor.values.resize(outputTensor.values.size() * lweSize);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      key.encrypt(&outputTensor.values[i * lweSize], inputTensor.values[i],
                  variance, *csprng);
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getSeededEncryptionTransformer(
    ClientKeyset keyset,
    const Message<concreteprotocol::LweCiphertextEncryptionInfo> &info) {

  auto key = keyset.lweSecretKeys[info.asReader().getKeyId()];
  auto lweDimension = info.asReader().getLweDimension();
  auto variance = info.asReader().getVariance();

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    // 3 = 2 (seed) + 1 (encrypted scalar)
    auto const ciphertextSize = 3;
    outputTensor.dimensions.push_back(ciphertextSize);
    outputTensor.values.resize(outputTensor.values.size() * ciphertextSize);
    struct Uint128 seed;
    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      csprng::getRandomSeed(&seed);
      // Write seed
      csprng::writeSeed(seed, &outputTensor.values[i * 3]);
      // Encrypt
      concrete_cpu_encrypt_seeded_lwe_ciphertext_u64(
          key.getBufferConst().data(), &outputTensor.values[i * 3 + 2],
          inputTensor.values[i], lweDimension, seed, variance);
    }
    return Value{outputTensor};
  };
}

Result<Transformer> getEncryptionSimulationTransformer(
    const Message<concreteprotocol::LweCiphertextEncryptionInfo> &info,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng) {

  auto lweDimension = info.asReader().getLweDimension();

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);

    for (size_t i = 0; i < inputTensor.values.size(); i++) {
      outputTensor.values[i] = sim_encrypt_lwe_u64(
          inputTensor.values[i], lweDimension, (void *)(*csprng).ptr);
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getDecryptionTransformer(
    ClientKeyset keyset,
    const Message<concreteprotocol::LweCiphertextEncryptionInfo> &info) {

  auto key = keyset.lweSecretKeys[info.asReader().getKeyId()];
  auto lweDimension = info.asReader().getLweDimension();
  auto lweSize = lweDimension + 1;

  return [=](Value input) {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.pop_back();
    outputTensor.values.resize(outputTensor.values.size() / lweSize);

    for (size_t i = 0; i < outputTensor.values.size(); i++) {
      key.decrypt(outputTensor.values[i], &inputTensor.values[i * lweSize]);
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getDecryptionSimulationTransformer() {
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
      uint64_t carry = output % 2;
      uint64_t mod = 1 << 3;
      output = ((output >> 1) + carry) % mod;
      outputTensor.values[i] = output;
    }

    return Value{outputTensor};
  };
}

Result<Transformer> getIntegerEncodingTransformer(
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  if (info.asReader().getMode().hasNative()) {
    return getNativeModeIntegerEncodingTransformer(info);
  } else if (info.asReader().getMode().hasChunked()) {
    return getChunkedModeIntegerEncodingTransformer(info);
  } else if (info.asReader().getMode().hasCrt()) {
    return getCrtModeIntegerEncodingTransformer(info);
  } else {
    return StringError(
        "Tried to construct integer encoding transformer without mode.");
  }
}

Result<Transformer> getIntegerDecodingTransformer(
    const Message<concreteprotocol::IntegerCiphertextEncodingInfo> &info) {
  if (info.asReader().getMode().hasNative()) {
    return getNativeModeIntegerDecodingTransformer(info);
  } else if (info.asReader().getMode().hasChunked()) {
    return getChunkedModeIntegerDecodingTransformer(info);
  } else if (info.asReader().getMode().hasCrt()) {
    return getCrtModeIntegerDecodingTransformer(info);
  } else {
    return StringError(
        "Tried to construct integer decoding transformer without mode.");
  }
}

Result<InputTransformer> TransformerFactory::getIndexInputTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasIndex()) {
    return StringError(
        "Tried to get index input transformer from non-index gate info.");
  }
  OUTCOME_TRY(auto verify, getIndexInputValueVerifier(gateInfo));
  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(verify(val));
    if (val.isSigned()) {
      val = val.toUnsigned();
    }
    auto output = val.intoRawTransportValue();
    output.asBuilder().initTypeInfo().setIndex(
        gateInfo.asReader().getTypeInfo().getIndex());
    return output;
  };
}

Result<OutputTransformer> TransformerFactory::getIndexOutputTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasIndex()) {
    return StringError(
        "Tried to get index output transformer from non-index gate info.");
  }
  OUTCOME_TRY(auto verify, getTransportValueVerifier(gateInfo));
  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(verify(transportVal));
    return Value::fromRawTransportValue(transportVal);
  };
}

Result<ArgTransformer> TransformerFactory::getIndexArgTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasIndex()) {
    return StringError(
        "Tried to get index arg transformer from non-index gate info.");
  }
  // The arg transformer is the same as the output transformer here ...
  return getIndexOutputTransformer(std::move(gateInfo));
}

Result<ReturnTransformer> TransformerFactory::getIndexReturnTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasIndex()) {
    return StringError(
        "Tried to get index return transformer from non-index gate info.");
  }
  // The return transformer is the same as the input transformer here ...
  return getIndexInputTransformer(std::move(gateInfo));
}

Result<InputTransformer> TransformerFactory::getPlaintextInputTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return StringError("Tried to get plaintext input transformer from "
                       "non-plaintext gate info.");
  }
  OUTCOME_TRY(auto verify, getPlaintextInputValueVerifier(gateInfo));
  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(verify(val));
    if (val.isSigned()) {
      val = val.toUnsigned();
    }
    auto output = val.intoRawTransportValue();
    output.asBuilder().initTypeInfo().setPlaintext(
        gateInfo.asReader().getTypeInfo().getPlaintext());
    return output;
  };
}

Result<OutputTransformer> TransformerFactory::getPlaintextOutputTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return StringError("Tried to get plaintext output transformer from "
                       "non-plaintext gate info.");
  }
  OUTCOME_TRY(auto verify, getTransportValueVerifier(gateInfo));
  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(verify(transportVal));
    return Value::fromRawTransportValue(transportVal);
  };
}

Result<ArgTransformer> TransformerFactory::getPlaintextArgTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return StringError("Tried to get plaintext arg transformer from "
                       "non-plaintext gate info.");
  }
  // The arg transformer is the same as the output transformer here ...
  return getPlaintextOutputTransformer(std::move(gateInfo));
}

Result<ReturnTransformer> TransformerFactory::getPlaintextReturnTransformer(
    Message<concreteprotocol::GateInfo> gateInfo) {
  if (!gateInfo.asReader().getTypeInfo().hasPlaintext()) {
    return StringError("Tried to get plaintext return transformer from "
                       "non-plaintext gate info.");
  }
  // The return transformer is the same as the input transformer here ...
  return getPlaintextInputTransformer(std::move(gateInfo));
}

Result<InputTransformer>
getLweCiphertextInputTransformer(Message<concreteprotocol::GateInfo> gateInfo,
                                 Transformer encryptionTransformer) {
  if (!gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return StringError("Tried to get lwe ciphertext input transformer from "
                       "non-ciphertext gate info.");
  }

  /// Generating the encoding transformer.
  Transformer encodingTransformer;
  if (gateInfo.asReader()
          .getTypeInfo()
          .getLweCiphertext()
          .getEncoding()
          .hasBoolean()) {
    OUTCOME_TRY(encodingTransformer, getBooleanEncodingTransformer());
  } else if (gateInfo.asReader()
                 .getTypeInfo()
                 .getLweCiphertext()
                 .getEncoding()
                 .hasInteger()) {
    OUTCOME_TRY(encodingTransformer,
                getIntegerEncodingTransformer(gateInfo.asReader()
                                                  .getTypeInfo()
                                                  .getLweCiphertext()
                                                  .getEncoding()
                                                  .getInteger()));
  } else {
    return StringError("Malformed gate info");
  }

  OUTCOME_TRY(auto verify, getLweCiphertextInputValueVerifier(gateInfo));
  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(verify(val));
    auto output =
        encryptionTransformer(encodingTransformer(val)).intoRawTransportValue();
    output.asBuilder().initTypeInfo().setLweCiphertext(
        gateInfo.asReader().getTypeInfo().getLweCiphertext());
    return output;
  };
}

Result<InputTransformer> TransformerFactory::getLweCiphertextInputTransformer(
    ClientKeyset keyset, Message<concreteprotocol::GateInfo> gateInfo,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng, bool useSimulation) {
  if (!useSimulation) {
    auto keyid = gateInfo.asReader()
                     .getTypeInfo()
                     .getLweCiphertext()
                     .getEncryption()
                     .getKeyId();
    if (keyid >= keyset.lweSecretKeys.size()) {
      return StringError(
          "Tried to generate lwe ciphertext input transformer with "
          "key id unavailable");
    }
  }
  /// Generating the encryption transformer.
  Transformer encryptionTransformer;
  if (useSimulation) {
    OUTCOME_TRY(encryptionTransformer,
                getEncryptionSimulationTransformer(gateInfo.asReader()
                                                       .getTypeInfo()
                                                       .getLweCiphertext()
                                                       .getEncryption(),
                                                   csprng));
  } else {
    auto compression =
        gateInfo.asReader().getTypeInfo().getLweCiphertext().getCompression();
    if (compression == concreteprotocol::Compression::NONE) {
      OUTCOME_TRY(encryptionTransformer,
                  getEncryptionTransformer(keyset,
                                           gateInfo.asReader()
                                               .getTypeInfo()
                                               .getLweCiphertext()
                                               .getEncryption(),
                                           csprng));
    } else if (compression == concreteprotocol::Compression::SEED) {
      OUTCOME_TRY(encryptionTransformer,
                  getSeededEncryptionTransformer(keyset, gateInfo.asReader()
                                                             .getTypeInfo()
                                                             .getLweCiphertext()
                                                             .getEncryption()));
    } else {
      return StringError(
          "Only none compression is currently supported for lwe ciphertext "
          "currently.");
    }
  }
  return concretelang::transformers::getLweCiphertextInputTransformer(
      gateInfo, encryptionTransformer);
}

Result<InputTransformer> TransformerFactory::getLweCiphertextInputTransformer(
    keysets::ClientPublicKeyset keyset,
    Message<concreteprotocol::GateInfo> gateInfo,
    std::shared_ptr<concretelang::csprng::SecretCSPRNG> csprng) {
  auto encryptionInfo =
      gateInfo.asReader().getTypeInfo().getLweCiphertext().getEncryption();
  OUTCOME_TRY(auto encryptionTransformer,
              getEncryptionTransformer(keyset, encryptionInfo, csprng));
  return concretelang::transformers::getLweCiphertextInputTransformer(
      gateInfo, encryptionTransformer);
}

Result<Transformer> getSeededLweCiphertextDecompressionTransformer(
    const Message<concreteprotocol::LweCiphertextEncryptionInfo> &info) {

  auto lweDimension = info.asReader().getLweDimension();
  auto lweSize = lweDimension + 1;
  return [=](Value input) -> Value {
    auto inputTensor = input.getTensor<uint64_t>().value();
    auto outputTensor = Tensor<uint64_t>(inputTensor);
    outputTensor.dimensions.back() = lweSize;
    auto size = 1;
    for (auto d : outputTensor.dimensions) {
      size *= d;
    }
    outputTensor.values.resize(size);

    for (size_t i = 0; i < inputTensor.values.size(); i += 3) {
      Uint128 seed;
      csprng::readSeed(seed, &inputTensor.values[i]);
      concrete_cpu_decompress_seeded_lwe_ciphertext_u64(
          &outputTensor.values[(i / 3) * lweSize], &inputTensor.values[i + 2],
          lweDimension, seed);
    }
    return Value{outputTensor};
  };
}

Result<ArgTransformer> TransformerFactory::getLweCiphertextArgTransformer(
    Message<concreteprotocol::GateInfo> gateInfo, bool useSimulation) {
  if (!gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return StringError("Tried to get lwe ciphertext arg transformer from "
                       "non-ciphertext gate info.");
  }

  /// Generating the decompression transformer.
  Transformer decompressionTransformer;
  auto lweCiphertextInfo = gateInfo.asReader().getTypeInfo().getLweCiphertext();
  auto compression = lweCiphertextInfo.getCompression();
  if (compression == concreteprotocol::Compression::NONE || useSimulation) {
    OUTCOME_TRY(decompressionTransformer, getNoneDecompressionTransformer());
  } else if (compression == concreteprotocol::Compression::SEED) {
    OUTCOME_TRY(decompressionTransformer,
                getSeededLweCiphertextDecompressionTransformer(
                    lweCiphertextInfo.getEncryption()));
  } else {
    return StringError(
        "Only none compression is currently supported for lwe ciphertext "
        "currently.");
  }

  // Generating the verifier.
  TransportValueVerifier verify;
  if (useSimulation) {
    OUTCOME_TRY(verify, getObliviousTransportValueVerifier());
  } else {
    OUTCOME_TRY(verify, getTransportValueVerifier(gateInfo));
  }

  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(verify(transportVal));
    return decompressionTransformer(Value::fromRawTransportValue(transportVal));
  };
}

Result<ReturnTransformer> TransformerFactory::getLweCiphertextReturnTransformer(
    Message<concreteprotocol::GateInfo> gateInfo, bool useSimulation) {
  if (!gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return StringError("Tried to get lwe ciphertext return transformer from "
                       "non-ciphertext gate info.");
  }

  /// Generating the compression transformer.
  Transformer compressionTransformer;
  if (gateInfo.asReader().getTypeInfo().getLweCiphertext().getCompression() ==
      concreteprotocol::Compression::NONE) {
    OUTCOME_TRY(compressionTransformer, getNoneCompressionTransformer());
  } else {
    return StringError(
        "Only none compression is currently supported for lwe ciphertext "
        "currently.");
  }

  // Generating the verifier.
  ValueVerifier verify;
  if (useSimulation) {
    OUTCOME_TRY(verify, getObliviousValueVerifier());
  } else {
    OUTCOME_TRY(verify, getLweCiphertextOutputValueVerifier(gateInfo));
  }

  return [=](Value val) -> Result<TransportValue> {
    OUTCOME_TRYV(verify(val));
    auto output = compressionTransformer(val).intoRawTransportValue();
    output.asBuilder().initTypeInfo().setLweCiphertext(
        gateInfo.asReader().getTypeInfo().getLweCiphertext());
    return output;
  };
}

Result<OutputTransformer> TransformerFactory::getLweCiphertextOutputTransformer(
    ClientKeyset keyset, Message<concreteprotocol::GateInfo> gateInfo,
    bool useSimulation) {
  if (!gateInfo.asReader().getTypeInfo().hasLweCiphertext()) {
    return StringError("Tried to get lwe ciphertext output transformer from "
                       "non-ciphertext gate info.");
  }
  if (!useSimulation) {
    auto keyid = gateInfo.asReader()
                     .getTypeInfo()
                     .getLweCiphertext()
                     .getEncryption()
                     .getKeyId();
    if (keyid >= keyset.lweSecretKeys.size()) {
      return StringError(
          "Tried to generate lwe ciphertext output transformer with "
          "key id unavailable");
    }
  }

  /// Generating the decompression transformer.
  Transformer decompressionTransformer;
  if (gateInfo.asReader().getTypeInfo().getLweCiphertext().getCompression() ==
      concreteprotocol::Compression::NONE) {
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
                getDecryptionTransformer(keyset, gateInfo.asReader()
                                                     .getTypeInfo()
                                                     .getLweCiphertext()
                                                     .getEncryption()));
  }

  /// Generating the decoding transformer.
  Transformer decodingTransformer;
  if (gateInfo.asReader()
          .getTypeInfo()
          .getLweCiphertext()
          .getEncoding()
          .hasBoolean()) {
    OUTCOME_TRY(decodingTransformer, getBooleanDecodingTransformer());
  } else if (gateInfo.asReader()
                 .getTypeInfo()
                 .getLweCiphertext()
                 .getEncoding()
                 .hasInteger()) {
    OUTCOME_TRY(decodingTransformer,
                getIntegerDecodingTransformer(gateInfo.asReader()
                                                  .getTypeInfo()
                                                  .getLweCiphertext()
                                                  .getEncoding()
                                                  .getInteger()));
  } else {
    return StringError("Malformed gate info");
  }

  // Generating the verifier.
  TransportValueVerifier verify;
  if (useSimulation) {
    OUTCOME_TRY(verify, getObliviousTransportValueVerifier());
  } else {
    OUTCOME_TRY(verify, getTransportValueVerifier(gateInfo));
  }

  return [=](TransportValue transportVal) -> Result<Value> {
    OUTCOME_TRYV(verify(transportVal));
    return decodingTransformer(decryptionTransformer(
        decompressionTransformer(Value::fromRawTransportValue(transportVal))));
  };
}

} // namespace transformers
} // namespace concretelang

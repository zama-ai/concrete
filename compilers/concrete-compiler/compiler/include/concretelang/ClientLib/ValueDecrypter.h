// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_VALUE_DECRYPTER_H
#define CONCRETELANG_CLIENTLIB_VALUE_DECRYPTER_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <variant>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

outcome::checked<std::vector<uint64_t>, StringError>
decrypt_no_decode_one_result(const KeySet &keySet, uint pos, CircuitGate gate,
                             uint ct_count,
                             ScalarOrTensorOrCompressedData &buffers,
                             const ClientParameters &clientParameters);

outcome::checked<std::vector<uint64_t>, StringError>
decode_one_result(std::vector<uint64_t> decrypted_vector, Encoding encoding);

outcome::checked<std::vector<uint64_t>, StringError>
decrypt_decode_one_result(const KeySet &keySet, uint pos, CircuitGate gate,
                          uint ct_count,
                          ScalarOrTensorOrCompressedData &buffers,
                          const ClientParameters &clientParameters);

uint64_t decode_1padded_integer(uint64_t decrypted, uint64_t precision,
                                bool signe);

class ValueDecrypterInterface {
public:
  virtual outcome::checked<std::vector<uint64_t>, StringError>
  decryptValues(ScalarOrTensorOrCompressedData &value, size_t argPos) = 0;

  virtual ~ValueDecrypterInterface() = default;
};

/// @brief allows to transform a serializable value into a clear value
class ValueDecrypter : public ValueDecrypterInterface {
public:
  ValueDecrypter(KeySet &keySet, ClientParameters clientParameters)
      : _keySet(keySet), _clientParameters(clientParameters) {}

  outcome::checked<std::vector<uint64_t>, StringError>
  decryptValues(ScalarOrTensorOrCompressedData &value, size_t argPos) override {
    OUTCOME_TRY(auto gate, _clientParameters.ouput(argPos));

    if (!gate.isEncrypted()) {

      if (std::holds_alternative<ScalarData>(value)) {
        std::vector<uint64_t> result;

        result.push_back(std::get<ScalarData>(value).getValueAsU64());

        return result;
      }
#ifdef OUTPUT_COMPRESSION_SUPPORT
      else if (std::holds_alternative<
                   std::shared_ptr<comp::CompressedCiphertext>>(value)) {
        exit(1);
      }
#endif
      else {
        assert(std::holds_alternative<TensorData>(value));
        return std::get<TensorData>(value).asFlatVector<uint64_t>();
      }
    }

    uint ct_count = 1;

    for (uint i = 0; i < gate.shape.dimensions.size(); i++) {
      ct_count *= gate.shape.dimensions[i];
    }
    return decrypt_decode_one_result(_keySet, argPos, gate, ct_count, value,
                                     _clientParameters);
  };

  outcome::checked<CircuitGate, StringError> outputGate(size_t argPos) {
    return _clientParameters.ouput(argPos);
  }

  outcome::checked<std::vector<int64_t>, StringError> getShape(size_t pos) {
    OUTCOME_TRY(auto gate, outputGate(pos));
    return gate.shape.dimensions;
  }

private:
  KeySet &_keySet;
  ClientParameters _clientParameters;
};

class SimulatedValueDecrypter : public ValueDecrypterInterface {
public:
  SimulatedValueDecrypter(ClientParameters clientParameters)
      : _clientParameters(clientParameters) {}

  // TODO: a lot of this logic can be factorized when moving
  // `KeySet::decrypt_lwe` into the LWE ValueDecyrpter
  outcome::checked<std::vector<uint64_t>, StringError>
  decryptValues(ScalarOrTensorOrCompressedData &value, size_t argPos) override {
    OUTCOME_TRY(auto gate, _clientParameters.ouput(argPos));

    if (!gate.isEncrypted()) {
      if (std::holds_alternative<ScalarData>(value)) {
        std::vector<uint64_t> result;

        result.push_back(std::get<ScalarData>(value).getValueAsU64());

        return result;
      }

      assert(std::holds_alternative<TensorData>(value));

      return std::get<TensorData>(value).asFlatVector<uint64_t>();
    }

    uint ct_count = 1;

    for (uint i = 0; i < gate.shape.dimensions.size(); i++) {
      ct_count *= gate.shape.dimensions[i];
    }

    if (std::holds_alternative<ScalarData>(value)) {
      std::vector<uint64_t> result;

      result.push_back(std::get<ScalarData>(value).getValueAsU64());

      return decode_one_result(result, gate.encryption->encoding);
    }

    assert(std::holds_alternative<TensorData>(value));

    return decode_one_result(
        std::get<TensorData>(value).asFlatVector<uint64_t>(),
        gate.encryption->encoding);
  }

  outcome::checked<CircuitGate, StringError> outputGate(size_t argPos) {
    return _clientParameters.ouput(argPos);
  }

  outcome::checked<std::vector<int64_t>, StringError> getShape(size_t pos) {
    OUTCOME_TRY(auto gate, outputGate(pos));
    return gate.shape.dimensions;
  }

private:
  ClientParameters _clientParameters;
};

} // namespace clientlib
} // namespace concretelang

#endif

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_VALUE_DECRYPTER_H
#define CONCRETELANG_CLIENTLIB_VALUE_DECRYPTER_H

#include <iostream>

#include "boost/outcome.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class ValueDecrypterInterface {
protected:
  virtual outcome::checked<uint64_t, StringError>
  decryptValue(size_t argPos, uint64_t *ciphertext) = 0;
  /// Size of the low-level ciphertext, taking into account the CRT if used
  virtual int64_t ciphertextSize(CircuitGate &gate) = 0;
  /// Output gate at position `argPos`
  virtual outcome::checked<CircuitGate, StringError>
  outputGate(size_t argPos) = 0;
  /// Whether the value decrypter is simulating encryption
  virtual bool isSimulated() = 0;

  /// Whether argument at pos `argPos` is encrypted or not
  virtual outcome::checked<bool, StringError> isEncrypted(size_t argPos) {
    OUTCOME_TRY(auto gate, outputGate(argPos));
    return gate.isEncrypted();
  }

public:
  virtual ~ValueDecrypterInterface() = default;

  /// @brief Transforms a FHE value into a clear scalar value
  /// @tparam T The type of the clear scalar value
  /// @param value The value to decrypt
  /// @param pos The position of the argument
  /// @return Either the decrypted value or an error if the gate doesn't match
  /// the expected result.
  template <typename T>
  outcome::checked<T, StringError> decrypt(ScalarOrTensorData &value,
                                           size_t pos) {
    OUTCOME_TRY(auto encrypted, isEncrypted(pos));
    if (!encrypted)
      return std::get<clientlib::ScalarData>(value).getValue<T>();

    if (isSimulated()) {
      // value is a scalar in simulation
      auto ciphertext =
          std::get<clientlib::ScalarData>(value).getValue<uint64_t>();
      OUTCOME_TRY(auto decrypted, decryptValue(pos, &ciphertext));
      return (T)decrypted;
    }

    auto &buffer = std::get<clientlib::TensorData>(value);

    auto ciphertext = buffer.getOpaqueElementPointer(0);

    // Convert to uint64_t* as required by `KeySet::decrypt_lwe`
    // FIXME: this may break alignment restrictions on some
    // architectures
    auto ciphertextu64 = reinterpret_cast<uint64_t *>(ciphertext);
    OUTCOME_TRY(auto decrypted, decryptValue(pos, ciphertextu64));

    return (T)decrypted;
  }

  /// @brief Transforms a FHE value  into a vector of clear value
  /// @tparam T The type of the clear scalar value
  /// @param value The value to decrypt
  /// @param pos The position of the argument
  /// @return Either the decrypted value or an error if the gate doesn't match
  /// the expected result.
  template <typename T>
  outcome::checked<std::vector<T>, StringError>
  decryptTensor(ScalarOrTensorData &value, size_t pos) {
    OUTCOME_TRY(auto encrypted, isEncrypted(pos));
    if (!encrypted)
      return std::get<clientlib::TensorData>(value).asFlatVector<T>();

    auto &buffer = std::get<clientlib::TensorData>(value);
    OUTCOME_TRY(auto gate, outputGate(pos));
    auto lweSize = ciphertextSize(gate);

    std::vector<T> decryptedValues(buffer.length() / lweSize);
    for (size_t i = 0; i < decryptedValues.size(); i++) {
      auto ciphertext = buffer.getOpaqueElementPointer(i * lweSize);

      // Convert to uint64_t* as required by `KeySet::decrypt_lwe`
      // FIXME: this may break alignment restrictions on some
      // architectures
      auto ciphertextu64 = reinterpret_cast<uint64_t *>(ciphertext);
      OUTCOME_TRY(auto decrypted, decryptValue(pos, ciphertextu64));
      decryptedValues[i] = decrypted;
    }
    return decryptedValues;
  }

  /// Return the shape of the clear tensor of a result.
  outcome::checked<std::vector<int64_t>, StringError> getShape(size_t pos) {
    OUTCOME_TRY(auto gate, outputGate(pos));
    return gate.shape.dimensions;
  }
};

/// @brief allows to transform a serializable value into a clear value
class ValueDecrypter : public ValueDecrypterInterface {
public:
  ValueDecrypter(KeySet &keySet, ClientParameters clientParameters)
      : _keySet(keySet), _clientParameters(clientParameters) {}

protected:
  outcome::checked<uint64_t, StringError>
  decryptValue(size_t argPos, uint64_t *ciphertext) override {
    uint64_t decrypted;
    OUTCOME_TRYV(_keySet.decrypt_lwe(0, ciphertext, decrypted));
    return decrypted;
  };

  bool isSimulated() override { return false; }

  outcome::checked<CircuitGate, StringError>
  outputGate(size_t argPos) override {
    return _clientParameters.ouput(argPos);
  }

  int64_t ciphertextSize(CircuitGate &gate) override {
    return _clientParameters.lweBufferSize(gate);
  }

private:
  KeySet &_keySet;
  ClientParameters _clientParameters;
};

class SimulatedValueDecrypter : public ValueDecrypterInterface {
public:
  SimulatedValueDecrypter(ClientParameters clientParameters)
      : _clientParameters(clientParameters) {}

protected:
  // TODO: a lot of this logic can be factorized when moving
  // `KeySet::decrypt_lwe` into the LWE ValueDecyrpter
  outcome::checked<uint64_t, StringError>
  decryptValue(size_t argPos, uint64_t *ciphertext) override {
    uint64_t output;
    OUTCOME_TRY(auto gate, outputGate(argPos));
    auto encoding = gate.encryption->encoding;
    auto precision = encoding.precision;
    auto crtVec = gate.encryption->encoding.crt;
    if (crtVec.empty()) {
      output = *ciphertext;
      output >>= (64 - precision - 2);
      auto carry = output % 2;
      uint64_t mod = (((uint64_t)1) << (precision + 1));
      output = ((output >> 1) + carry) % mod;
      // Further decode signed integers.
      if (encoding.isSigned) {
        uint64_t maxPos = (((uint64_t)1) << (precision - 1));
        if (output >= maxPos) { // The output is actually negative.
          // Set the preceding bits to zero
          output |= UINT64_MAX << precision;
          // This makes sure when the value is cast to int64, it has the correct
          // value
        };
      }
    } else {
      // Decrypt and decode remainders
      std::vector<int64_t> remainders;
      for (auto modulus : crtVec) {
        output = *ciphertext;
        auto plaintext = crt::decode(output, modulus);
        remainders.push_back(plaintext);
        // each ciphertext is a scalar
        ciphertext = ciphertext + 1;
      }
      output = crt::iCrt(crtVec, remainders);
      // Further decode signed integers
      if (encoding.isSigned) {
        uint64_t maxPos = 1;
        for (auto prime : crtVec) {
          maxPos *= prime;
        }
        maxPos /= 2;
        if (output >= maxPos) {
          output -= maxPos * 2;
        }
      }
    }
    return output;
  }

  bool isSimulated() override { return true; }

  outcome::checked<CircuitGate, StringError>
  outputGate(size_t argPos) override {
    return _clientParameters.ouput(argPos);
  }

  /// @brief Ciphertext size in simulation
  /// When using CRT encoding, it's the number of blocks, otherwise, it's just 1
  /// scalar
  /// @param gate
  /// @return number of scalars to represent one input
  int64_t ciphertextSize(CircuitGate &gate) override {
    // ciphertext in simulation are only scalars
    assert(gate.encryption.has_value());
    auto crtSize = gate.encryption->encoding.crt.size();
    return crtSize == 0 ? 1 : crtSize;
  }

private:
  ClientParameters _clientParameters;
};

} // namespace clientlib
} // namespace concretelang

#endif

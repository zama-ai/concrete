// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_VALUE_EXPORTER_H
#define CONCRETELANG_CLIENTLIB_VALUE_EXPORTER_H

#include <ostream>

#include "boost/outcome.h"

#include "concretelang/ClientLib/CRT.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/simulation.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class ValueExporterInterface {
protected:
  virtual outcome::checked<void, StringError> encryptValue(CircuitGate &gate,
                                                           size_t argPos,
                                                           uint64_t *ciphertext,
                                                           uint64_t input) = 0;
  /// Encrypt and export a 64bits integer to a serializale value
  virtual outcome::checked<ScalarOrTensorData, StringError>
  exportEncryptValue(uint64_t arg, CircuitGate &gate, size_t argPos) = 0;
  /// Shape of the low-level buffer
  virtual std::vector<int64_t> bufferShape(CircuitGate &gate) = 0;
  /// Size of the low-level ciphertext, taking into account the CRT if used
  virtual int64_t ciphertextSize(CircuitGate &gate) = 0;
  /// Input gate at position `argPos`
  virtual outcome::checked<CircuitGate, StringError>
  inputGate(size_t argPos) = 0;

public:
  virtual ~ValueExporterInterface() = default;

  /// @brief Export a scalar 64 bits integer to a concreteprotocol::Value
  /// @param arg An 64 bits integer
  /// @param argPos The position of the argument to export
  /// @return Either the exported value ready to be sent to the server or an
  /// error if the gate doesn't match the expected argument.
  outcome::checked<ScalarOrTensorData, StringError> exportValue(uint64_t arg,
                                                                size_t argPos) {
    OUTCOME_TRY(auto gate, inputGate(argPos));
    if (gate.shape.size != 0) {
      return StringError("argument #") << argPos << " is not a scalar";
    }
    if (gate.encryption.has_value()) {
      return exportEncryptValue(arg, gate, argPos);
    }
    return exportClearValue(arg);
  }

  /// @brief Export a tensor like buffer of values to a serializable value
  /// @tparam T The type of values hold by the buffer
  /// @param arg A pointer to a memory area where the values are stored
  /// @param shape The shape of the tensor
  /// @param argPos The position of the argument to export
  /// @return Either the exported value ready to be sent to the server or an
  /// error if the gate doesn't match the expected argument.
  template <typename T>
  outcome::checked<ScalarOrTensorData, StringError>
  exportValue(const T *arg, llvm::ArrayRef<int64_t> shape, size_t argPos) {
    OUTCOME_TRY(auto gate, inputGate(argPos));
    OUTCOME_TRYV(checkShape(shape, gate.shape, argPos));
    if (gate.encryption.has_value()) {
      return exportEncryptTensor(arg, shape, gate, argPos);
    }
    return exportClearTensor(arg, shape, gate);
  }

protected:
  /// Export a 64bits integer to a serializable value
  virtual outcome::checked<ScalarOrTensorData, StringError>
  exportClearValue(uint64_t arg) {
    return ScalarData(arg);
  }

  /// Export a tensor like buffer to a serializable value
  template <typename T>
  outcome::checked<ScalarOrTensorData, StringError>
  exportClearTensor(const T *arg, llvm::ArrayRef<int64_t> shape,
                    CircuitGate &gate) {
    auto bitsPerValue = bitWidthAsWord(gate.shape.width);
    auto sizes = bufferShape(gate);
    TensorData td(sizes, bitsPerValue, gate.shape.sign);
    llvm::ArrayRef<T> values(arg, TensorData::getNumElements(sizes));
    td.bulkAssign(values);
    return std::move(td);
  }

  /// Export and encrypt a tensor like buffer to a serializable value
  template <typename T>
  outcome::checked<ScalarOrTensorData, StringError>
  exportEncryptTensor(const T *arg, llvm::ArrayRef<int64_t> shape,
                      CircuitGate &gate, size_t argPos) {
    // Create and allocate the TensorData that will holds encrypted values
    auto sizes = bufferShape(gate);
    TensorData td(sizes, EncryptedScalarElementType,
                  EncryptedScalarElementWidth);

    // Iterate over values and encrypt at the right place the value
    auto lweSize = ciphertextSize(gate);
    for (size_t i = 0, offset = 0; i < gate.shape.size;
         i++, offset += lweSize) {
      OUTCOME_TRYV(encryptValue(
          gate, argPos, td.getElementPointer<uint64_t>(offset), arg[i]));
    }
    return std::move(td);
  }

private:
  static outcome::checked<void, StringError>
  checkShape(llvm::ArrayRef<int64_t> shape, CircuitGateShape expected,
             size_t argPos) {
    // Check the shape of tensor
    if (expected.dimensions.empty()) {
      return StringError("argument #") << argPos << "is not a tensor";
    }
    if (shape.size() != expected.dimensions.size()) {
      return StringError("argument #")
             << argPos << "has not the expected number of dimension, got "
             << shape.size() << " expected " << expected.dimensions.size();
    }

    // Check shape
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] != expected.dimensions[i]) {
        return StringError("argument #")
               << argPos << " has not the expected dimension #" << i
               << " , got " << shape[i] << " expected "
               << expected.dimensions[i];
      }
    }
    return outcome::success();
  }
};

/// @brief The ArgumentsExporter allows to transform clear
/// arguments to the one expected by a server lambda.
class ValueExporter : public ValueExporterInterface {

public:
  /// @brief
  /// @param keySet
  /// @param clientParameters
  // TODO: Get rid of the reference here could make troubles (see for KeySet
  // copy constructor or shared pointers)
  ValueExporter(KeySet &keySet, ClientParameters clientParameters)
      : _keySet(keySet), _clientParameters(clientParameters) {}

protected:
  outcome::checked<void, StringError> encryptValue(CircuitGate &gate,
                                                   size_t argPos,
                                                   uint64_t *ciphertext,
                                                   uint64_t input) override {
    return _keySet.encode_encrypt_lwe(argPos, ciphertext, input);
  }

  outcome::checked<CircuitGate, StringError> inputGate(size_t argPos) override {
    return _clientParameters.input(argPos);
  }

  std::vector<int64_t> bufferShape(CircuitGate &gate) override {
    return _clientParameters.bufferShape(gate);
  }

  int64_t ciphertextSize(CircuitGate &gate) override {
    return _clientParameters.lweBufferSize(gate);
  }

  /// Encrypt and export a 64bits integer to a serializale value
  outcome::checked<ScalarOrTensorData, StringError>
  exportEncryptValue(uint64_t arg, CircuitGate &gate, size_t argPos) override {
    std::vector<int64_t> shape = _clientParameters.bufferShape(gate);

    // Create and allocate the TensorData that will holds encrypted value
    TensorData td(shape, clientlib::EncryptedScalarElementType,
                  clientlib::EncryptedScalarElementWidth);

    // Encrypt the value
    OUTCOME_TRYV(
        encryptValue(gate, argPos, td.getElementPointer<uint64_t>(0), arg));
    return std::move(td);
  }

private:
  KeySet &_keySet;
  ClientParameters _clientParameters;
};

/// @brief The SimulatedValueExporter allows to transform clear
/// arguments to the one expected by a server lambda during simulation.
class SimulatedValueExporter : public ValueExporterInterface {

public:
  SimulatedValueExporter(ClientParameters clientParameters)
      : _clientParameters(clientParameters), csprng(0) {}

protected:
  outcome::checked<void, StringError> encryptValue(CircuitGate &gate,
                                                   size_t argPos,
                                                   uint64_t *ciphertext,
                                                   uint64_t input) override {
    auto crtVec = gate.encryption->encoding.crt;
    OUTCOME_TRY(auto skParam, _clientParameters.lweSecretKeyParam(gate));
    auto lwe_dim = skParam.lweDimension();
    if (crtVec.empty()) {
      auto precision = gate.encryption->encoding.precision;
      auto encoded_input = input << (64 - (precision + 1));
      *ciphertext =
          sim_encrypt_lwe_u64(encoded_input, lwe_dim, (void *)csprng.ptr);
    } else {
      // Put each decomposition into a new ciphertext
      auto product = concretelang::clientlib::crt::productOfModuli(crtVec);
      for (auto modulus : crtVec) {
        auto plaintext = crt::encode(input, modulus, product);
        *ciphertext =
            sim_encrypt_lwe_u64(plaintext, lwe_dim, (void *)csprng.ptr);
        // each ciphertext is a scalar
        ciphertext += 1;
      }
    }
    return outcome::success();
  }

  /// Simulate encrypt and export a 64bits integer to a serializale value
  outcome::checked<ScalarOrTensorData, StringError>
  exportEncryptValue(uint64_t arg, CircuitGate &gate, size_t argPos) override {
    uint64_t encValue = 0;
    OUTCOME_TRYV(encryptValue(gate, argPos, &encValue, arg));
    return ScalarData(encValue);
  }

  outcome::checked<CircuitGate, StringError> inputGate(size_t argPos) override {
    return _clientParameters.input(argPos);
  }

  std::vector<int64_t> bufferShape(CircuitGate &gate) override {
    return _clientParameters.bufferShape(gate, true);
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
  concretelang::clientlib::ConcreteCSPRNG csprng;
};

} // namespace clientlib
} // namespace concretelang

#endif

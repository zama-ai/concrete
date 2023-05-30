// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_ENCRYPTED_ARGS_H
#define CONCRETELANG_CLIENTLIB_ENCRYPTED_ARGS_H

#include <ostream>

#include "boost/outcome.h"

#include "../Common/Error.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/Types.h"
#include "concretelang/Common/BitsSize.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class PublicArguments;

/// @brief The ArgumentsExporter allows to transform clear
/// arguments to the one expected by a server lambda.
class ValueExporter {

public:
  /// @brief
  /// @param keySet
  /// @param clientParameters
  // TODO: Get rid of the reference here could make troubles (see for KeySet
  // copy constructor or shared pointers)
  ValueExporter(KeySet &keySet, ClientParameters clientParameters)
      : _keySet(keySet), _clientParameters(clientParameters) {}

  /// @brief Export a scalar 64 bits integer to a concreteprocol::Value
  /// @param arg An 64 bits integer
  /// @param argPos The position of the argument to export
  /// @return Either the exported value ready to be sent to the server or an
  /// error if the gate doesn't match the expected argument.
  outcome::checked<ScalarOrTensorData, StringError> exportValue(uint64_t arg,
                                                                size_t argPos) {
    OUTCOME_TRY(auto gate, _clientParameters.input(argPos));
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
    OUTCOME_TRY(auto gate, _clientParameters.input(argPos));
    OUTCOME_TRYV(checkShape(shape, gate.shape, argPos));
    if (gate.encryption.has_value()) {
      return exportEncryptTensor(arg, shape, gate, argPos);
    }
    return exportClearTensor(arg, shape, gate);
  }

private:
  /// Export a 64bits integer to a serializable value
  outcome::checked<ScalarOrTensorData, StringError>
  exportClearValue(uint64_t arg) {
    return ScalarData(arg);
  }

  /// Encrypt and export a 64bits integer to a serializale value
  outcome::checked<ScalarOrTensorData, StringError>
  exportEncryptValue(uint64_t arg, CircuitGate &gate, size_t argPos) {
    std::vector<int64_t> shape = _clientParameters.bufferShape(gate);

    // Create and allocate the TensorData that will holds encrypted value
    TensorData td(shape, clientlib::EncryptedScalarElementType,
                  clientlib::EncryptedScalarElementWidth);

    // Encrypt the value
    OUTCOME_TRYV(
        _keySet.encrypt_lwe(argPos, td.getElementPointer<uint64_t>(0), arg));
    return std::move(td);
  }

  /// Export a tensor like buffer to a serializable value
  template <typename T>
  outcome::checked<ScalarOrTensorData, StringError>
  exportClearTensor(const T *arg, llvm::ArrayRef<int64_t> shape,
                    CircuitGate &gate) {
    auto bitsPerValue = bitWidthAsWord(gate.shape.width);
    auto sizes = _clientParameters.bufferShape(gate);
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
    auto sizes = _clientParameters.bufferShape(gate);
    TensorData td(sizes, EncryptedScalarElementType,
                  EncryptedScalarElementWidth);

    // Iterate over values and encrypt at the right place the value
    auto lweSize = _clientParameters.lweBufferSize(gate);
    for (size_t i = 0, offset = 0; i < gate.shape.size;
         i++, offset += lweSize) {
      OUTCOME_TRYV(_keySet.encrypt_lwe(
          argPos, td.getElementPointer<uint64_t>(offset), arg[i]));
    }
    return std::move(td);
  }

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

private:
  KeySet &_keySet;
  ClientParameters _clientParameters;
};

/// Temporary object used to hold and encrypt parameters before calling a
/// ClientLambda. Use preferably TypeClientLambda and serializeCall(Args...).
/// Otherwise convert it to a PublicArguments and use
/// serializeCall(PublicArguments, KeySet).
class EncryptedArguments {

public:
  EncryptedArguments() {}

  /// Encrypts args thanks the given KeySet and pack the encrypted arguments
  /// to an EncryptedArguments
  template <typename... Args>
  static outcome::checked<std::unique_ptr<EncryptedArguments>, StringError>
  create(KeySet &keySet, Args... args) {
    auto encryptedArgs = std::make_unique<EncryptedArguments>();
    OUTCOME_TRYV(encryptedArgs->pushArgs(keySet, args...));
    return std::move(encryptedArgs);
  }

  template <typename ArgT>
  static outcome::checked<std::unique_ptr<EncryptedArguments>, StringError>
  create(KeySet &keySet, const llvm::ArrayRef<ArgT> args) {
    auto encryptedArgs = EncryptedArguments::empty();
    for (size_t i = 0; i < args.size(); i++) {
      OUTCOME_TRYV(encryptedArgs->pushArg(args[i], keySet));
    }
    OUTCOME_TRYV(encryptedArgs->checkAllArgs(keySet));
    return std::move(encryptedArgs);
  }

  static std::unique_ptr<EncryptedArguments> empty() {
    return std::make_unique<EncryptedArguments>();
  }

  /// Export encrypted arguments as public arguments, reset the encrypted
  /// arguments, i.e. move all buffers to the PublicArguments and reset the
  /// positional counter.
  outcome::checked<std::unique_ptr<PublicArguments>, StringError>
  exportPublicArguments(ClientParameters clientParameters);

  /// Check that all arguments as been pushed.
  // TODO: Remove public method here
  outcome::checked<void, StringError> checkAllArgs(KeySet &keySet);

public:
  /// Add a uint64_t scalar argument.
  outcome::checked<void, StringError> pushArg(uint64_t arg, KeySet &keySet) {
    ValueExporter exporter(keySet, keySet.clientParameters());
    OUTCOME_TRY(auto value, exporter.exportValue(arg, values.size()));
    values.push_back(std::move(value));
    return outcome::success();
  }

  /// Add a vector-tensor argument.
  outcome::checked<void, StringError> pushArg(std::vector<uint8_t> arg,
                                              KeySet &keySet) {
    return pushArg((uint8_t *)arg.data(),
                   llvm::ArrayRef<int64_t>{(int64_t)arg.size()}, keySet);
  }

  /// Add a 1D tensor argument with data and size of the dimension.
  template <typename T>
  outcome::checked<void, StringError> pushArg(const T *data, int64_t dim1,
                                              KeySet &keySet) {
    return pushArg(std::vector<uint8_t>(data, data + dim1), keySet);
  }

  /// Add a 1D tensor argument.
  template <size_t size>
  outcome::checked<void, StringError> pushArg(std::array<uint8_t, size> arg,
                                              KeySet &keySet) {
    return pushArg((uint8_t *)arg.data(), llvm::ArrayRef<int64_t>{size},
                   keySet);
  }

  /// Add a 2D tensor argument.
  template <size_t size0, size_t size1>
  outcome::checked<void, StringError>
  pushArg(std::array<std::array<uint8_t, size1>, size0> arg, KeySet &keySet) {
    return pushArg((uint8_t *)arg.data(), llvm::ArrayRef<int64_t>{size0, size1},
                   keySet);
  }

  /// Add a 3D tensor argument.
  template <size_t size0, size_t size1, size_t size2>
  outcome::checked<void, StringError>
  pushArg(std::array<std::array<std::array<uint8_t, size2>, size1>, size0> arg,
          KeySet &keySet) {
    return pushArg((uint8_t *)arg.data(),
                   llvm::ArrayRef<int64_t>{size0, size1, size2}, keySet);
  }

  // Generalize by computing shape by template recursion

  /// Set a argument at the given pos as a 1D tensor of T.
  template <typename T>
  outcome::checked<void, StringError> pushArg(T *data, int64_t dim1,
                                              KeySet &keySet) {
    return pushArg<T>(data, llvm::ArrayRef<int64_t>(&dim1, 1), keySet);
  }

  /// Set a argument at the given pos as a tensor of T.
  template <typename T>
  outcome::checked<void, StringError>
  pushArg(T *data, llvm::ArrayRef<int64_t> shape, KeySet &keySet) {
    return pushArg(static_cast<const T *>(data), shape, keySet);
  }

  template <typename T>
  outcome::checked<void, StringError>
  pushArg(const T *data, llvm::ArrayRef<int64_t> shape, KeySet &keySet) {
    ValueExporter exporter(keySet, keySet.clientParameters());
    OUTCOME_TRY(auto value, exporter.exportValue(data, shape, values.size()));
    values.push_back(std::move(value));
    return outcome::success();
  }

  /// Recursive case for scalars: extract first scalar argument from
  /// parameter pack and forward rest
  template <typename Arg0, typename... OtherArgs>
  outcome::checked<void, StringError> pushArgs(KeySet &keySet, Arg0 arg0,
                                               OtherArgs... others) {
    OUTCOME_TRYV(pushArg(arg0, keySet));
    return pushArgs(keySet, others...);
  }

  /// Recursive case for tensors: extract pointer and size from
  /// parameter pack and forward rest
  template <typename Arg0, typename... OtherArgs>
  outcome::checked<void, StringError>
  pushArgs(KeySet &keySet, Arg0 *arg0, size_t size, OtherArgs... others) {
    OUTCOME_TRYV(pushArg(arg0, size, keySet));
    return pushArgs(keySet, others...);
  }

  /// Terminal case of pushArgs
  outcome::checked<void, StringError> pushArgs(KeySet &keySet) {
    return checkAllArgs(keySet);
  }

private:
  /// Store buffers of ciphertexts
  std::vector<ScalarOrTensorData> values;
};

} // namespace clientlib
} // namespace concretelang

#endif

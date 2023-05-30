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

#include "concrete-protocol.pb.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class PublicArguments;

/// @brief The ArgumentsExporter class is a class that allows to transform clear
/// arguments to the one expected by a server lambda.
class ValueExporter {

public:
  /// @brief
  /// @param keySet
  /// @param clientParameters TODO CircuitInfo here
  // TODO: Get rid of the reference here could make troubles (see for KeySet
  // copy constructor or shared pointers)
  ValueExporter(KeySet &keySet, ClientParameters clientParameters)
      : _keySet(keySet), _clientParameters(clientParameters) {}

  /// @brief Export a scalar 64 bits integer to a concreteprocol::Value
  /// @param arg An 64 bits integer
  /// @param argPos The position of the argument to export
  /// @return Either the exported value ready to be sent to the server or an
  /// error if the gate doesn't match the expected argument.
  outcome::checked<concreteprotocol::Value, StringError>
  exportValue(uint64_t arg, size_t argPos);

  /// @brief Export a tensor like buffer of values to a concreteprotocol::Value
  /// @tparam T The type of values hold by the buffer
  /// @param arg A pointer to a memory area where the values are stored
  /// @param shape The shape of the tensor
  /// @param argPos The position of the argument to export
  /// @return Either the exported value ready to be sent to the server or an
  /// error if the gate doesn't match the expected argument.
  template <typename T>
  outcome::checked<concreteprotocol::Value, StringError>
  exportValue(const T *arg, llvm::ArrayRef<int64_t> shape, size_t argPos) {
    OUTCOME_TRY(auto gate, _clientParameters.input(argPos));
    OUTCOME_TRY_V(checkShape(shape, gate.shape));
    if (gate.encryption.has_value()) {
      return encryptTensor(arg, shape, gate.encryption.value());
    }
    // Set sizes
    std::vector<int64_t> sizes = keySet.clientParameters().bufferShape(input);

    if (input.encryption.has_value()) {
      TensorData td(sizes, EncryptedScalarElementType,
                    EncryptedScalarElementWidth);

      auto lweSize = keySet.clientParameters().lweBufferSize(input);

      for (size_t i = 0, offset = 0; i < input.shape.size;
           i++, offset += lweSize) {
        OUTCOME_TRYV(keySet.encrypt_lwe(
            pos, td.getElementPointer<uint64_t>(offset), data[i]));
      }
      ciphertextBuffers.push_back(std::move(td));
    } else {
      auto bitsPerValue = bitWidthAsWord(input.shape.width);

      TensorData td(sizes, bitsPerValue, input.shape.sign);
      llvm::ArrayRef<T> values(data, TensorData::getNumElements(sizes));
      td.bulkAssign(values);
      ciphertextBuffers.push_back(std::move(td));
    }
  }

  template <typename T>
  static outcome::checked<concreteprotocol::Value, StringError>
  encryptTensor(const T *arg, llvm::ArrayRef<int64_t> shape,
                EncryptionGate &encryptionInfo) {
    return StringError("encryptTensor NYI");
  }

  // TODO: concreteprotocol::Shape
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
    return outcome::success()
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
  EncryptedArguments() : currentPos(0) {}

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
  outcome::checked<void, StringError> pushArg(uint64_t arg, KeySet &keySet);

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
    OUTCOME_TRYV(checkPushTooManyArgs(keySet));
    auto pos = currentPos;
    CircuitGate input = keySet.inputGate(pos);
    // Check the width of data
    if (input.shape.width > 64) {
      return StringError("argument #")
             << pos << " width > 64 bits is not supported";
    }
    // Check the shape of tensor
    if (input.shape.dimensions.empty()) {
      return StringError("argument #") << pos << "is not a tensor";
    }
    if (shape.size() != input.shape.dimensions.size()) {
      return StringError("argument #")
             << pos << "has not the expected number of dimension, got "
             << shape.size() << " expected " << input.shape.dimensions.size();
    }

    // Check shape
    for (size_t i = 0; i < shape.size(); i++) {
      if (shape[i] != input.shape.dimensions[i]) {
        return StringError("argument #")
               << pos << " has not the expected dimension #" << i << " , got "
               << shape[i] << " expected " << input.shape.dimensions[i];
      }
    }

    // Set sizes
    std::vector<int64_t> sizes = keySet.clientParameters().bufferShape(input);

    if (input.encryption.has_value()) {
      TensorData td(sizes, EncryptedScalarElementType,
                    EncryptedScalarElementWidth);

      auto lweSize = keySet.clientParameters().lweBufferSize(input);

      for (size_t i = 0, offset = 0; i < input.shape.size;
           i++, offset += lweSize) {
        OUTCOME_TRYV(keySet.encrypt_lwe(
            pos, td.getElementPointer<uint64_t>(offset), data[i]));
      }
      ciphertextBuffers.push_back(std::move(td));
    } else {
      auto bitsPerValue = bitWidthAsWord(input.shape.width);

      TensorData td(sizes, bitsPerValue, input.shape.sign);
      llvm::ArrayRef<T> values(data, TensorData::getNumElements(sizes));
      td.bulkAssign(values);
      ciphertextBuffers.push_back(std::move(td));
    }

    currentPos++;
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
  outcome::checked<void, StringError> checkPushTooManyArgs(KeySet &keySet);

private:
  /// Position of the next pushed argument
  size_t currentPos;

  /// Store buffers of ciphertexts
  std::vector<concreteprotocol::Value> values;
};

} // namespace clientlib
} // namespace concretelang

#endif

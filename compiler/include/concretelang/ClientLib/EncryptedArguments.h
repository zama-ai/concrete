// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
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

class EncryptedArguments {
  /// Temporary object used to hold and encrypt parameters before calling a
  /// ClientLambda. Use preferably TypeClientLambda and serializeCall(Args...).
  /// Otherwise convert it to a PublicArguments and use
  /// serializeCall(PublicArguments, KeySet).
public:
  EncryptedArguments() : currentPos(0) {}

  /// Encrypts args thanks the given KeySet and pack the encrypted arguments to
  /// an EncryptedArguments
  template <typename... Args>
  static outcome::checked<std::unique_ptr<EncryptedArguments>, StringError>
  create(std::shared_ptr<KeySet> keySet, Args... args) {
    auto arguments = std::make_unique<EncryptedArguments>();
    OUTCOME_TRYV(arguments->pushArgs(keySet, args...));
    return arguments;
  }

  /// Export encrypted arguments as public arguments, reset the encrypted
  /// arguments, i.e. move all buffers to the PublicArguments and reset the
  /// positional counter.
  outcome::checked<std::unique_ptr<PublicArguments>, StringError>
  exportPublicArguments(ClientParameters clientParameters,
                        RuntimeContext runtimeContext);

public:
  /// Add a uint8_t scalar argument.
  outcome::checked<void, StringError> pushArg(uint8_t arg,
                                              std::shared_ptr<KeySet> keySet);

  // Add a uint64_t scalar argument.
  outcome::checked<void, StringError> pushArg(uint64_t arg,
                                              std::shared_ptr<KeySet> keySet);

  /// Add a vector-tensor argument.
  outcome::checked<void, StringError> pushArg(std::vector<uint8_t> arg,
                                              std::shared_ptr<KeySet> keySet);

  /// Add a 1D tensor argument.
  template <size_t size>
  outcome::checked<void, StringError> pushArg(std::array<uint8_t, size> arg,
                                              std::shared_ptr<KeySet> keySet) {
    return pushArg(8, (void *)arg.data(), {size}, keySet);
  }

  /// Add a 2D tensor argument.
  template <size_t size0, size_t size1>
  outcome::checked<void, StringError>
  pushArg(std::array<std::array<uint8_t, size1>, size0> arg,
          std::shared_ptr<KeySet> keySet) {
    return pushArg(8, (void *)arg.data(), {size0, size1}, keySet);
  }

  /// Add a 3D tensor argument.
  template <size_t size0, size_t size1, size_t size2>
  outcome::checked<void, StringError>
  pushArg(std::array<std::array<std::array<uint8_t, size2>, size1>, size0> arg,
          std::shared_ptr<KeySet> keySet) {
    return pushArg(8, (void *)arg.data(), {size0, size1, size2}, keySet);
  }

  // Generalize by computing shape by template recursion

  // Set a argument at the given pos as a 1D tensor of T.
  template <typename T>
  outcome::checked<void, StringError> pushArg(T *data, size_t dim1,
                                              std::shared_ptr<KeySet> keySet) {
    return pushArg<T>(data, llvm::ArrayRef<size_t>(&dim1, 1), keySet);
  }

  // Set a argument at the given pos as a tensor of T.
  template <typename T>
  outcome::checked<void, StringError> pushArg(T *data,
                                              llvm::ArrayRef<int64_t> shape,
                                              std::shared_ptr<KeySet> keySet) {
    return pushArg(8 * sizeof(T), static_cast<void *>(data), shape, keySet);
  }

  outcome::checked<void, StringError> pushArg(size_t width, void *data,
                                              llvm::ArrayRef<int64_t> shape,
                                              std::shared_ptr<KeySet> keySet);

  /// Push a variadic list of arguments.
  template <typename Arg0, typename... OtherArgs>
  outcome::checked<void, StringError> pushArgs(std::shared_ptr<KeySet> keySet,
                                               Arg0 arg0, OtherArgs... others) {
    OUTCOME_TRYV(pushArg(arg0, keySet));
    return pushArgs(keySet, others...);
  }

  // Terminal case of pushArgs
  outcome::checked<void, StringError> pushArgs(std::shared_ptr<KeySet> keySet) {
    return checkAllArgs(keySet);
  }

private:
  outcome::checked<void, StringError>
  checkPushTooManyArgs(std::shared_ptr<KeySet> keySet);
  outcome::checked<void, StringError>
  checkAllArgs(std::shared_ptr<KeySet> keySet);

private:
  // Position of the next pushed argument
  size_t currentPos;
  std::vector<void *> preparedArgs;

  // Store buffers of ciphertexts
  std::vector<encrypted_scalars_and_sizes_t> ciphertextBuffers;
};

} // namespace clientlib
} // namespace concretelang

#endif

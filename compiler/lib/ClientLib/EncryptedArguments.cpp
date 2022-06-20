// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/EncryptedArguments.h"
#include "concretelang/ClientLib/PublicArguments.h"

namespace concretelang {
namespace clientlib {

using StringError = concretelang::error::StringError;

outcome::checked<std::unique_ptr<PublicArguments>, StringError>
EncryptedArguments::exportPublicArguments(ClientParameters clientParameters,
                                          RuntimeContext runtimeContext) {
  return std::make_unique<PublicArguments>(
      clientParameters, std::move(preparedArgs), std::move(ciphertextBuffers));
}

outcome::checked<void, StringError>
EncryptedArguments::pushArg(uint64_t arg, KeySet &keySet) {
  OUTCOME_TRYV(checkPushTooManyArgs(keySet));
  auto pos = currentPos++;
  OUTCOME_TRY(CircuitGate input, keySet.clientParameters().input(pos));
  if (input.shape.size != 0) {
    return StringError("argument #") << pos << " is not a scalar";
  }
  if (!input.encryption.hasValue()) {
    // clear scalar: just push the argument
    preparedArgs.push_back((void *)arg);
    return outcome::success();
  }
  // Allocate empty
  ciphertextBuffers.resize(ciphertextBuffers.size() + 1);
  TensorData &values_and_sizes = ciphertextBuffers.back();
  values_and_sizes.sizes = keySet.clientParameters().bufferShape(input);
  values_and_sizes.values.resize(keySet.clientParameters().bufferSize(input));
  OUTCOME_TRYV(keySet.encrypt_lwe(pos, values_and_sizes.values.data(), arg));
  // Note: Since we bufferized lwe ciphertext take care of memref calling
  // convention
  // allocated
  preparedArgs.push_back(nullptr);
  // aligned
  preparedArgs.push_back((void *)values_and_sizes.values.data());
  // offset
  preparedArgs.push_back((void *)0);
  // sizes
  for (auto size : values_and_sizes.sizes) {
    preparedArgs.push_back((void *)size);
  }
  // strides
  int64_t stride = values_and_sizes.length();
  for (size_t i = 0; i < values_and_sizes.sizes.size() - 1; i++) {
    auto size = values_and_sizes.sizes[i];
    stride = (size == 0 ? 0 : (stride / size));
    preparedArgs.push_back((void *)stride);
  }
  preparedArgs.push_back((void *)1);
  return outcome::success();
}

outcome::checked<void, StringError>
EncryptedArguments::checkPushTooManyArgs(KeySet &keySet) {
  size_t arity = keySet.numInputs();
  if (currentPos < arity) {
    return outcome::success();
  }
  return StringError("function has arity ")
         << arity << " but is applied to too many arguments";
}

outcome::checked<void, StringError>
EncryptedArguments::checkAllArgs(KeySet &keySet) {
  size_t arity = keySet.numInputs();
  if (currentPos == arity) {
    return outcome::success();
  }
  return StringError("function expects ")
         << arity << " arguments but has been called with " << currentPos
         << " arguments";
}

} // namespace clientlib
} // namespace concretelang

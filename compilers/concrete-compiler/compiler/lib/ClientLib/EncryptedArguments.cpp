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
EncryptedArguments::exportPublicArguments(ClientParameters clientParameters) {
  return std::make_unique<PublicArguments>(clientParameters,
                                           std::move(ciphertextBuffers));
}

/// Split the input integer into `size` chunks of `chunkWidth` bits each
std::vector<uint64_t> chunkInput(uint64_t value, size_t size,
                                 unsigned int chunkWidth) {
  std::vector<uint64_t> chunks;
  chunks.reserve(size);
  uint64_t mask = (1 << chunkWidth) - 1;
  for (size_t i = 0; i < size; i++) {
    auto chunk = value & mask;
    chunks.push_back((uint64_t)chunk);
    value >>= chunkWidth;
  }
  return chunks;
}

outcome::checked<void, StringError>
EncryptedArguments::pushArg(uint64_t arg, KeySet &keySet) {
  OUTCOME_TRYV(checkPushTooManyArgs(keySet));
  OUTCOME_TRY(CircuitGate input, keySet.clientParameters().input(currentPos));
  // a chunked input is represented as a tensor in lower levels, and need to to
  // splitted into chunks and encrypted as such
  if (input.chunkInfo.has_value()) {
    std::vector<uint64_t> chunks =
        chunkInput(arg, input.shape.size, input.chunkInfo.value().width);
    return this->pushArg(chunks.data(), input.shape.size, keySet);
  }
  // we only increment if we don't forward the call to another pushArg method
  auto pos = currentPos++;
  if (input.shape.size != 0) {
    return StringError("argument #") << pos << " is not a scalar";
  }
  if (!input.encryption.has_value()) {
    // clear scalar: just push the argument
    ciphertextBuffers.push_back(ScalarData(arg));
    return outcome::success();
  }

  std::vector<int64_t> shape = keySet.clientParameters().bufferShape(input);

  // Allocate empty
  ciphertextBuffers.emplace_back(
      TensorData(shape, clientlib::EncryptedScalarElementType,
                 clientlib::EncryptedScalarElementWidth));
  TensorData &values_and_sizes = ciphertextBuffers.back().getTensor();

  OUTCOME_TRYV(keySet.encrypt_lwe(
      pos, values_and_sizes.getElementPointer<decrypted_scalar_t>(0), arg));

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

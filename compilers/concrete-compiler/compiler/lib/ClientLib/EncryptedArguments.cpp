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
  auto values = std::vector<ScalarOrTensorData>();
  values.reserve(this->values.size());

  for (auto &&value : this->values) {
    values.push_back(std::move(value));
  }

  return std::make_unique<PublicArguments>(clientParameters, std::move(values));
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

outcome::checked<void, StringError> checkSizes(size_t actualSize,
                                               size_t expectedSize) {
  if (actualSize == expectedSize) {
    return outcome::success();
  }
  return StringError("function expects ")
         << expectedSize << " arguments but has been called with " << actualSize
         << " arguments";
}

outcome::checked<void, StringError>
EncryptedArguments::checkAllArgs(KeySet &keySet) {
  size_t arity = keySet.numInputs();
  return checkSizes(values.size(), arity);
}

outcome::checked<void, StringError>
EncryptedArguments::checkAllArgs(ClientParameters &params) {
  size_t arity = params.inputs.size();
  return checkSizes(values.size(), arity);
}

} // namespace clientlib
} // namespace concretelang

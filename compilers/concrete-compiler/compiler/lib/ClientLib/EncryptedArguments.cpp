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

outcome::checked<void, StringError>
EncryptedArguments::checkAllArgs(KeySet &keySet) {
  size_t arity = keySet.numInputs();
  if (values.size() == arity) {
    return outcome::success();
  }
  return StringError("function expects ")
         << arity << " arguments but has been called with " << values.size()
         << " arguments";
}

} // namespace clientlib
} // namespace concretelang

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_KEYSETCACHE_H_
#define CONCRETELANG_CLIENTLIB_KEYSETCACHE_H_

#include "concretelang/ClientLib/KeySet.h"

namespace concretelang {
namespace clientlib {

class KeySet;

class KeySetCache {
  std::string backingDirectoryPath;

public:
  KeySetCache(std::string backingDirectoryPath)
      : backingDirectoryPath(backingDirectoryPath) {}

  static outcome::checked<std::unique_ptr<KeySet>, StringError>
  generate(std::shared_ptr<KeySetCache> optionalCache, ClientParameters &params,
           uint64_t seed_msb, uint64_t seed_lsb);

  outcome::checked<std::unique_ptr<KeySet>, StringError>
  generate(ClientParameters &params, uint64_t seed_msb, uint64_t seed_lsb);

private:
  static outcome::checked<std::unique_ptr<KeySet>, StringError>
  loadKeys(ClientParameters &params, uint64_t seed_msb, uint64_t seed_lsb,
           std::string folderPath);

  outcome::checked<std::unique_ptr<KeySet>, StringError>
  loadOrGenerateSave(ClientParameters &params, uint64_t seed_msb,
                     uint64_t seed_lsb);
};

} // namespace clientlib
} // namespace concretelang

#endif

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_SUPPORT_KEYSETCACHE_H_
#define CONCRETELANG_SUPPORT_KEYSETCACHE_H_

#include "concretelang/Support/KeySet.h"

namespace mlir {
namespace concretelang {

class KeySet;

class KeySetCache {
  std::string backingDirectoryPath;

public:
  KeySetCache(std::string backingDirectoryPath)
      : backingDirectoryPath(backingDirectoryPath) {}

  llvm::Expected<std::unique_ptr<KeySet>>
  tryLoadOrGenerateSave(ClientParameters &params, uint64_t seed_msb,
                        uint64_t seed_lsb);

private:
  static llvm::Expected<std::unique_ptr<KeySet>>
  tryLoadKeys(ClientParameters &params, uint64_t seed_msb, uint64_t seed_lsb,
              llvm::SmallString<0> &folderPath);
};

} // namespace concretelang
} // namespace mlir

#endif
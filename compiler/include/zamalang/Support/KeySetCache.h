// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_SUPPORT_KEYSETCACHE_H_
#define ZAMALANG_SUPPORT_KEYSETCACHE_H_

#include "zamalang/Support/KeySet.h"

namespace mlir {
namespace zamalang {

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

} // namespace zamalang
} // namespace mlir

#endif
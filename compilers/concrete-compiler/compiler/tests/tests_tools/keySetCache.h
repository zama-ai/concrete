#ifndef TEST_TOOLS_KEYSETCACHE_H
#define TEST_TOOLS_KEYSETCACHE_H

#include "llvm/Support/Path.h"

#include "concretelang/ClientLib/KeySetCache.h"

static inline std::optional<concretelang::clientlib::KeySetCache>
getTestKeySetCache() {

  llvm::SmallString<0> cachePath;
  llvm::sys::path::system_temp_directory(true, cachePath);
  llvm::sys::path::append(cachePath, "KeySetCache");

  auto cachePathStr = std::string(cachePath);

  std::cout << "Using KeySetCache dir: " << cachePathStr << "\n";

  return std::optional<concretelang::clientlib::KeySetCache>(
      concretelang::clientlib::KeySetCache(cachePathStr));
}

static inline std::shared_ptr<concretelang::clientlib::KeySetCache>
getTestKeySetCachePtr() {
  return std::make_shared<concretelang::clientlib::KeySetCache>(
      getTestKeySetCache().value());
}
#endif

#ifndef TEST_TOOLS_KEYSETCACHE_H
#define TEST_TOOLS_KEYSETCACHE_H

#include "concretelang/ClientLib/KeySetCache.h"
#include "llvm/Support/Path.h"

#ifdef CONCRETELANG_TEST_KEYCACHE_PATH
#define CACHE_PATH CONCRETELANG_TEST_KEYCACHE_PATH
#else
#define CACHE_PATH "KeySetCache"
#endif

static inline std::optional<concretelang::clientlib::KeySetCache>
getTestKeySetCache() {

  llvm::SmallString<0> cachePath;
  llvm::sys::path::system_temp_directory(true, cachePath);
  llvm::sys::path::append(cachePath, CACHE_PATH);

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

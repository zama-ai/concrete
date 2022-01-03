// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "boost/outcome.h"

#include "concretelang/ClientLib/KeySetCache.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <sstream>
#include <string>

extern "C" {
#include "concrete-ffi.h"
}

namespace concretelang {
namespace clientlib {

using StringError = concretelang::error::StringError;

static std::string readFile(llvm::SmallString<0> &path) {
  std::ifstream in((std::string)path, std::ofstream::binary);
  std::stringstream sbuffer;
  sbuffer << in.rdbuf();
  return sbuffer.str();
}

static void writeFile(llvm::SmallString<0> &path, Buffer content) {
  std::ofstream out((std::string)path, std::ofstream::binary);
  out.write((const char *)content.pointer, content.length);
  out.close();
}

LweSecretKey_u64 *loadSecretKey(llvm::SmallString<0> &path) {
  std::string content = readFile(path);
  BufferView buffer = {(const uint8_t *)content.c_str(), content.length()};
  return deserialize_lwe_secret_key_u64(buffer);
}

LweKeyswitchKey_u64 *loadKeyswitchKey(llvm::SmallString<0> &path) {
  std::string content = readFile(path);
  BufferView buffer = {(const uint8_t *)content.c_str(), content.length()};
  return deserialize_lwe_keyswitching_key_u64(buffer);
}

LweBootstrapKey_u64 *loadBootstrapKey(llvm::SmallString<0> &path) {
  std::string content = readFile(path);
  BufferView buffer = {(const uint8_t *)content.c_str(), content.length()};
  return deserialize_lwe_bootstrap_key_u64(buffer);
}

void saveSecretKey(llvm::SmallString<0> &path, LweSecretKey_u64 *key) {
  Buffer buffer = serialize_lwe_secret_key_u64(key);
  writeFile(path, buffer);
  free(buffer.pointer);
}

void saveBootstrapKey(llvm::SmallString<0> &path, LweBootstrapKey_u64 *key) {
  Buffer buffer = serialize_lwe_bootstrap_key_u64(key);
  writeFile(path, buffer);
  free(buffer.pointer);
}

void saveKeyswitchKey(llvm::SmallString<0> &path, LweKeyswitchKey_u64 *key) {
  Buffer buffer = serialize_lwe_keyswitching_key_u64(key);
  writeFile(path, buffer);
  free(buffer.pointer);
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySetCache::loadKeys(ClientParameters &params, uint64_t seed_msb,
                      uint64_t seed_lsb, std::string folderPath) {
  // TODO: text dump of all parameter in /hash
  auto key_set = std::make_unique<KeySet>();

  std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
      secretKeys;
  std::map<LweSecretKeyID, std::pair<BootstrapKeyParam, LweBootstrapKey_u64 *>>
      bootstrapKeys;
  std::map<LweSecretKeyID, std::pair<KeyswitchKeyParam, LweKeyswitchKey_u64 *>>
      keyswitchKeys;

  // Load LWE secret keys
  for (auto secretKeyParam : params.secretKeys) {
    auto id = secretKeyParam.first;
    auto param = secretKeyParam.second;
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "secretKey_" + id);
    LweSecretKey_u64 *sk = loadSecretKey(path);
    secretKeys[id] = {param, sk};
  }
  // Load bootstrap keys
  for (auto bootstrapKeyParam : params.bootstrapKeys) {
    auto id = bootstrapKeyParam.first;
    auto param = bootstrapKeyParam.second;
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "pbsKey_" + id);
    LweBootstrapKey_u64 *bsk = loadBootstrapKey(path);
    bootstrapKeys[id] = {param, bsk};
  }
  // Load keyswitch keys
  for (auto keyswitchParam : params.keyswitchKeys) {
    auto id = keyswitchParam.first;
    auto param = keyswitchParam.second;
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "ksKey_" + id);
    LweKeyswitchKey_u64 *ksk = loadKeyswitchKey(path);
    keyswitchKeys[id] = {param, ksk};
  }

  key_set->setKeys(secretKeys, bootstrapKeys, keyswitchKeys);

  OUTCOME_TRYV(key_set->setupEncryptionMaterial(params, seed_msb, seed_lsb));

  return std::move(key_set);
}

outcome::checked<void, StringError> saveKeys(KeySet &key_set,
                                             llvm::SmallString<0> &folderPath) {
  llvm::SmallString<0> folderIncompletePath = folderPath;

  folderIncompletePath.append(".incomplete");

  auto err = llvm::sys::fs::create_directories(folderIncompletePath);
  if (err) {
    return StringError("Cannot create directory \"")
           << std::string(folderIncompletePath) << "\": " << err.message();
  }

  // Save LWE secret keys
  for (auto secretKeyParam : key_set.getSecretKeys()) {
    auto id = secretKeyParam.first;
    auto key = secretKeyParam.second.second;
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "secretKey_" + id);
    saveSecretKey(path, key);
  }
  // Save bootstrap keys
  for (auto bootstrapKeyParam : key_set.getBootstrapKeys()) {
    auto id = bootstrapKeyParam.first;
    auto key = bootstrapKeyParam.second.second;
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "pbsKey_" + id);
    saveBootstrapKey(path, key);
  }
  // Save keyswitch keys
  for (auto keyswitchParam : key_set.getKeyswitchKeys()) {
    auto id = keyswitchParam.first;
    auto key = keyswitchParam.second.second;
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "ksKey_" + id);
    saveKeyswitchKey(path, key);
  }

  err = llvm::sys::fs::rename(folderIncompletePath, folderPath);
  if (err) {
    llvm::sys::fs::remove_directories(folderIncompletePath);
  }
  if (!llvm::sys::fs::exists(folderPath)) {
    return StringError("Cannot save directory \"")
           << std::string(folderPath) << "\"";
  }

  return outcome::success();
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySetCache::loadOrGenerateSave(ClientParameters &params, uint64_t seed_msb,
                                uint64_t seed_lsb) {

  llvm::SmallString<0> folderPath =
      llvm::SmallString<0>(this->backingDirectoryPath);

  llvm::sys::path::append(folderPath, std::to_string(params.hash()));

  llvm::sys::path::append(folderPath, std::to_string(seed_msb) + "_" +
                                          std::to_string(seed_lsb));

  if (llvm::sys::fs::exists(folderPath)) {
    return loadKeys(params, seed_msb, seed_lsb, std::string(folderPath));
  }

  // Creating a lock for concurrent generation
  llvm::SmallString<0> lockPath(folderPath);
  lockPath.append("lock");
  int FD_lock;
  llvm::sys::fs::create_directories(llvm::sys::path::parent_path(lockPath));
  // Open or create the lock file
  auto err = llvm::sys::fs::openFile(
      lockPath, FD_lock, llvm::sys::fs::CreationDisposition::CD_OpenAlways,
      llvm::sys::fs::FileAccess::FA_Write, llvm::sys::fs::OpenFlags::OF_None);

  if (err) {
    // parent does not exists OR right issue (creation or write)
    return StringError("Cannot access \"")
           << std::string(lockPath) << "\": " << err.message();
  }

  // The first to lock will generate while the others waits
  llvm::sys::fs::lockFile(FD_lock);
  auto cleanUnlock = llvm::make_scope_exit([&]() {
    llvm::sys::fs::unlockFile(FD_lock);
    llvm::sys::fs::remove(lockPath);
  });

  if (llvm::sys::fs::exists(folderPath)) {
    // Others returns here
    return loadKeys(params, seed_msb, seed_lsb, std::string(folderPath));
  }

  OUTCOME_TRY(auto key_set, KeySet::generate(params, seed_msb, seed_lsb));

  OUTCOME_TRYV(saveKeys(*key_set, folderPath));

  return std::move(key_set);
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySetCache::generate(std::shared_ptr<KeySetCache> cache,
                      ClientParameters &params, uint64_t seed_msb,
                      uint64_t seed_lsb) {
  return cache ? cache->loadOrGenerateSave(params, seed_msb, seed_lsb)
               : KeySet::generate(params, seed_msb, seed_lsb);
}

} // namespace clientlib
} // namespace concretelang

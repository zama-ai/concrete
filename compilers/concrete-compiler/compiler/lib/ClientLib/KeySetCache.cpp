// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifdef OUTPUT_COMPRESSION_SUPPORT
#include "CompressLWE/defines.h"
#endif
#include "boost/outcome.h"

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/ClientLib/Serializers.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"

#include <fstream>
#include <sstream>
#include <string>
#include <utime.h>

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
inline void getApproval() {
  std::cerr << "DANGER: You are using an empty unsecure secret keys. Enter "
               "\"y\" to continue: ";
  char answer;
  std::cin >> answer;
  if (answer != 'y') {
    std::abort();
  }
}
#endif

namespace concretelang {
namespace clientlib {

using StringError = concretelang::error::StringError;

template <class Key>
outcome::checked<Key, StringError> loadKey(llvm::SmallString<0> &path,
                                           Key(deser)(std::istream &istream)) {

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif
  std::ifstream in((std::string)path, std::ofstream::binary);
  if (in.fail()) {
    return StringError("Cannot access " + (std::string)path);
  }
  auto key = deser(in);
  if (in.bad()) {
    return StringError("Cannot load key at path(") << (std::string)path << ")";
  }
  return key;
}

template <class Key>
outcome::checked<void, StringError> saveKey(llvm::SmallString<0> &path,
                                            Key &key) {
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif
  std::ofstream out((std::string)path, std::ofstream::binary);
  if (out.fail()) {
    return StringError("Cannot access " + (std::string)path);
  }
  out << key;
  if (out.bad()) {
    return StringError("Cannot save key at path(") << (std::string)path << ")";
  }
  out.close();
  return outcome::success();
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySetCache::loadKeys(ClientParameters &params, uint64_t seed_msb,
                      uint64_t seed_lsb, std::string folderPath) {
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif

  // Mark the folder as recently use.
  // e.g. so the CI can do some cleanup of unused keys.
  utime(folderPath.c_str(), nullptr);

  std::vector<LweSecretKey> secretKeys;
  std::vector<LweBootstrapKey> bootstrapKeys;
  std::vector<LweKeyswitchKey> keyswitchKeys;
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
#ifdef OUTPUT_COMPRESSION_SUPPORT
  std::optional<PaiFullKeys> paiFullKeys;
#endif

  // Load secret keys
  for (auto p : llvm::enumerate(params.secretKeys)) {
    // TODO - Check parameters?
    // auto param = secretKeyParam.second;
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "secretKey_" + std::to_string(p.index()));
    OUTCOME_TRY(auto key, loadKey(path, readLweSecretKey));
    secretKeys.push_back(key);
  }
  // Load bootstrap keys
  for (auto p : llvm::enumerate(params.bootstrapKeys)) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "pbsKey_" + std::to_string(p.index()));
    OUTCOME_TRY(auto key, loadKey(path, readLweBootstrapKey));
    bootstrapKeys.push_back(key);
  }
  // Load keyswitch keys
  for (auto p : llvm::enumerate(params.keyswitchKeys)) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "ksKey_" + std::to_string(p.index()));
    OUTCOME_TRY(auto key, loadKey(path, readLweKeyswitchKey));
    keyswitchKeys.push_back(key);
  }

  for (auto p : llvm::enumerate(params.packingKeyswitchKeys)) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "pksKey_" + std::to_string(p.index()));
    OUTCOME_TRY(auto key, loadKey(path, readPackingKeyswitchKey));
    packingKeyswitchKeys.push_back(key);
  }

#ifdef OUTPUT_COMPRESSION_SUPPORT
  if (params.paiCompKeys.has_value()) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "pai_comp_Key");
    OUTCOME_TRY(auto key, loadKey(path, readPaiFullKey));
    paiFullKeys = std::move(key);
  }
#endif

  __uint128_t seed = seed_msb;
  seed <<= 64;
  seed += seed_lsb;

  auto csprng = ConcreteCSPRNG(seed);

#ifdef OUTPUT_COMPRESSION_SUPPORT
  OUTCOME_TRY(auto keySet,
              KeySet::fromKeys(params, secretKeys, bootstrapKeys, keyswitchKeys,
                               packingKeyswitchKeys, std::move(paiFullKeys),
                               std::move(csprng)));
#else
  OUTCOME_TRY(auto keySet,
              KeySet::fromKeys(params, secretKeys, bootstrapKeys, keyswitchKeys,
                               packingKeyswitchKeys, std::move(csprng)));
#endif

  return std::move(keySet);
}

outcome::checked<void, StringError> saveKeys(KeySet &key_set,
                                             llvm::SmallString<0> &folderPath) {
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif

  llvm::SmallString<0> folderIncompletePath = folderPath;

  folderIncompletePath.append(".incomplete");

  auto err = llvm::sys::fs::create_directories(folderIncompletePath);
  if (err) {
    return StringError("Cannot create directory \"")
           << std::string(folderIncompletePath) << "\": " << err.message();
  }
  // Save LWE secret keys
  for (auto p : llvm::enumerate(key_set.getSecretKeys())) {

    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "secretKey_" + std::to_string(p.index()));
    OUTCOME_TRYV(saveKey(path, p.value()));
  }
  // Save bootstrap keys
  for (auto p : llvm::enumerate(key_set.getBootstrapKeys())) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "pbsKey_" + std::to_string(p.index()));
    OUTCOME_TRYV(saveKey(path, p.value()));
  }
  // Save keyswitch keys
  for (auto p : llvm::enumerate(key_set.getKeyswitchKeys())) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "ksKey_" + std::to_string(p.index()));
    OUTCOME_TRYV(saveKey(path, p.value()));
  }
  // Save packing keyswitch keys
  for (auto p : llvm::enumerate(key_set.getPackingKeyswitchKeys())) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "pksKey_" + std::to_string(p.index()));
    OUTCOME_TRYV(saveKey(path, p.value()));
  }

#ifdef OUTPUT_COMPRESSION_SUPPORT
  // Save compression keys
  if (key_set.getPaiFullKey().has_value()) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(path, "pai_comp_Key");
    OUTCOME_TRYV(saveKey(path, *key_set.getPaiFullKey()));
  }
#endif

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

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif

  llvm::SmallString<0> folderPath =
      llvm::SmallString<0>(this->backingDirectoryPath);

  llvm::sys::path::append(folderPath, std::to_string(params.hash()));

  llvm::sys::path::append(folderPath, std::to_string(seed_msb) + "_" +
                                          std::to_string(seed_lsb));

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

  // The lock is released when the function returns.
  // => any intermediate state in the function is not visible to others.
  auto unlockAtReturn = llvm::make_scope_exit([&]() {
    llvm::sys::fs::closeFile(FD_lock);
    llvm::sys::fs::unlockFile(FD_lock);
    llvm::sys::fs::remove(lockPath);
  });
  llvm::sys::fs::lockFile(FD_lock);

  if (llvm::sys::fs::exists(folderPath)) {
    // Once it has been generated by another process (or was alread here)
    auto keys = loadKeys(params, seed_msb, seed_lsb, std::string(folderPath));
    if (keys.has_value()) {
      return keys;
    } else {
      std::cerr << std::string(keys.error().mesg) << "\n";
      std::cerr << "Invalid KeySetCache entry " << std::string(folderPath)
                << "\n";
      llvm::sys::fs::remove_directories(folderPath);
      // Then we can continue as it didn't exist
    }
  }

  std::cerr << "KeySetCache: miss, regenerating " << std::string(folderPath)
            << "\n";

  __uint128_t seed = seed_msb;
  seed <<= 64;
  seed += seed_lsb;

  auto csprng = ConcreteCSPRNG(seed);

  OUTCOME_TRY(auto key_set, KeySet::generate(params, std::move(csprng)));

  OUTCOME_TRYV(saveKeys(*key_set, folderPath));

  return std::move(key_set);
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySetCache::generate(std::shared_ptr<KeySetCache> cache,
                      ClientParameters &params, uint64_t seed_msb,
                      uint64_t seed_lsb) {
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif

  __uint128_t seed = seed_msb;
  seed <<= 64;
  seed += seed_lsb;

  auto csprng = ConcreteCSPRNG(seed);
  return cache ? cache->loadOrGenerateSave(params, seed_msb, seed_lsb)
               : KeySet::generate(params, std::move(csprng));
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySetCache::generate(ClientParameters &params, uint64_t seed_msb,
                      uint64_t seed_lsb) {
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif

  return loadOrGenerateSave(params, seed_msb, seed_lsb);
}

} // namespace clientlib
} // namespace concretelang

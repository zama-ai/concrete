// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Keysets.h"
#include "capnp/message.h"
#include "concrete-cpu.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keys.h"
#include "kj/common.h"
#include "kj/io.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <errno.h>
#include <fcntl.h>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <unistd.h>
#include <utime.h>

using concretelang::csprng::EncryptionCSPRNG;
using concretelang::csprng::SecretCSPRNG;
using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::keys::LweBootstrapKey;
using concretelang::keys::LweKeyswitchKey;
using concretelang::keys::LweSecretKey;
using concretelang::keys::PackingKeyswitchKey;

/// The default reading limit of capnp must be increased for large keys.
const capnp::ReaderOptions KEY_READER_OPTS =
    capnp::ReaderOptions{7000000000, 64};

namespace concretelang {
namespace keysets {

ClientKeyset
ClientKeyset::fromProto(const Message<concreteprotocol::ClientKeyset> &proto) {
  auto output = ClientKeyset();
  for (auto skProto : proto.asReader().getLweSecretKeys()) {
    output.lweSecretKeys.push_back(LweSecretKey::fromProto(skProto));
  }

  return output;
}

Message<concreteprotocol::ClientKeyset> ClientKeyset::toProto() const {
  auto output = Message<concreteprotocol::ClientKeyset>();
  output.asBuilder().initLweSecretKeys(lweSecretKeys.size());
  for (size_t i = 0; i < lweSecretKeys.size(); i++) {
    output.asBuilder().getLweSecretKeys().setWithCaveats(
        i, lweSecretKeys[i].toProto().asReader());
  }

  return output;
}

ServerKeyset
ServerKeyset::fromProto(const Message<concreteprotocol::ServerKeyset> &proto) {
  auto output = ServerKeyset();
  for (auto bskProto : proto.asReader().getLweBootstrapKeys()) {
    output.lweBootstrapKeys.push_back(LweBootstrapKey::fromProto(bskProto));
  }

  for (auto kskProto : proto.asReader().getLweKeyswitchKeys()) {
    output.lweKeyswitchKeys.push_back(LweKeyswitchKey::fromProto(kskProto));
  }

  for (auto pkskProto : proto.asReader().getPackingKeyswitchKeys()) {
    output.packingKeyswitchKeys.push_back(
        PackingKeyswitchKey::fromProto(pkskProto));
  }

  return output;
}

Message<concreteprotocol::ServerKeyset> ServerKeyset::toProto() const {
  auto output = Message<concreteprotocol::ServerKeyset>();
  output.asBuilder().initLweBootstrapKeys(lweBootstrapKeys.size());
  for (size_t i = 0; i < lweBootstrapKeys.size(); i++) {
    output.asBuilder().getLweBootstrapKeys().setWithCaveats(
        i, lweBootstrapKeys[i].toProto().asReader());
  }

  output.asBuilder().initLweKeyswitchKeys(lweKeyswitchKeys.size());
  for (size_t i = 0; i < lweKeyswitchKeys.size(); i++) {
    output.asBuilder().getLweKeyswitchKeys().setWithCaveats(
        i, lweKeyswitchKeys[i].toProto().asReader());
  }

  output.asBuilder().initPackingKeyswitchKeys(packingKeyswitchKeys.size());
  for (size_t i = 0; i < packingKeyswitchKeys.size(); i++) {
    output.asBuilder().getPackingKeyswitchKeys().setWithCaveats(
        i, packingKeyswitchKeys[i].toProto().asReader());
  }

  return output;
}

Keyset::Keyset(const Message<concreteprotocol::KeysetInfo> &info,
               SecretCSPRNG &secretCsprng, EncryptionCSPRNG &encryptionCsprng) {
  for (auto keyInfo : info.asReader().getLweSecretKeys()) {
    client.lweSecretKeys.push_back(LweSecretKey(keyInfo, secretCsprng));
  }
  for (auto keyInfo : info.asReader().getLweBootstrapKeys()) {
    server.lweBootstrapKeys.push_back(LweBootstrapKey(
        keyInfo, client.lweSecretKeys[keyInfo.getInputId()],
        client.lweSecretKeys[keyInfo.getOutputId()], encryptionCsprng));
  }
  for (auto keyInfo : info.asReader().getLweKeyswitchKeys()) {
    server.lweKeyswitchKeys.push_back(LweKeyswitchKey(
        keyInfo, client.lweSecretKeys[keyInfo.getInputId()],
        client.lweSecretKeys[keyInfo.getOutputId()], encryptionCsprng));
  }
  for (auto keyInfo : info.asReader().getPackingKeyswitchKeys()) {
    server.packingKeyswitchKeys.push_back(PackingKeyswitchKey(
        keyInfo, client.lweSecretKeys[keyInfo.getInputId()],
        client.lweSecretKeys[keyInfo.getOutputId()], encryptionCsprng));
  }
}

Keyset Keyset::fromProto(const Message<concreteprotocol::Keyset> &proto) {
  auto server = ServerKeyset::fromProto(proto.asReader().getServer());
  auto client = ClientKeyset::fromProto(proto.asReader().getClient());

  return {server, client};
}

Message<concreteprotocol::Keyset> Keyset::toProto() const {
  auto output = Message<concreteprotocol::Keyset>();
  auto serverProto = server.toProto();
  auto clientProto = client.toProto();
  output.asBuilder().setServer(serverProto.asReader());
  output.asBuilder().setClient(clientProto.asReader());
  return output;
}

template <typename ProtoKey>
Result<Message<ProtoKey>> loadKeyProto(std::string path) {
  std::ifstream in((std::string)path, std::ofstream::binary);
  if (in.fail()) {
    return StringError("Cannot load key at path " + (std::string)path +
                       " Error: " + strerror(errno));
  }
  Message<ProtoKey> keyBlob;
  OUTCOME_TRYV(keyBlob.readBinaryFromIstream(in, KEY_READER_OPTS));
  return keyBlob;
}

template <typename ProtoKey, typename Key>
Result<Key> loadKey(std::string path) {
  Message<ProtoKey> proto;
  OUTCOME_TRY(auto keyProto, loadKeyProto<ProtoKey>(path));
  return Key::fromProto(keyProto);
}

template <typename ProtoKey>
Result<void> saveKeyProto(Message<ProtoKey> keyProto, std::string path) {
  std::ofstream out((std::string)path, std::ofstream::binary);
  if (out.fail()) {
    return StringError("Cannot save key at path: " + (std::string)path +
                       " Error: " + strerror(errno));
  }
  OUTCOME_TRYV(keyProto.writeBinaryToOstream(out));
  return outcome::success();
}

template <typename ProtoKey, typename Key>
Result<void> saveKey(Key key, std::string path) {
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif
  auto proto = key.toProto();
  OUTCOME_TRYV(saveKeyProto<ProtoKey>(std::move(proto), path));
  return outcome::success();
}

Result<Keyset>
loadKeysFromFiles(const Message<concreteprotocol::KeysetInfo> &keysetInfo,
                  __uint128_t secret_seed, __uint128_t encryption_seed,
                  std::string folderPath) {
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

  // Load secret keys
  for (auto keyInfo : keysetInfo.asReader().getLweSecretKeys()) {
    // TODO - Check parameters?
    // auto param = secretKeyParam.second;
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path,
                            "secretKey_" + std::to_string(keyInfo.getId()));
    OUTCOME_TRY(auto key, loadKey<concreteprotocol::LweSecretKey, LweSecretKey>(
                              (std::string)path));
    secretKeys.push_back(key);
  }
  // Load bootstrap keys
  for (auto keyInfo : keysetInfo.asReader().getLweBootstrapKeys()) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "pbsKey_" + std::to_string(keyInfo.getId()));
    OUTCOME_TRY(auto key,
                loadKey<concreteprotocol::LweBootstrapKey, LweBootstrapKey>(
                    (std::string)path));
    bootstrapKeys.push_back(key);
  }
  // Load keyswitch keys
  for (auto keyInfo : keysetInfo.asReader().getLweKeyswitchKeys()) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "ksKey_" + std::to_string(keyInfo.getId()));
    OUTCOME_TRY(auto key,
                loadKey<concreteprotocol::LweKeyswitchKey, LweKeyswitchKey>(
                    (std::string)path));
    keyswitchKeys.push_back(key);
  }
  // Load packing keyswitch keys
  for (auto keyInfo : keysetInfo.asReader().getPackingKeyswitchKeys()) {
    // TODO - Check parameters?
    // auto param = p.value();
    llvm::SmallString<0> path(folderPath);
    llvm::sys::path::append(path, "pksKey_" + std::to_string(keyInfo.getId()));
    OUTCOME_TRY(
        auto key,
        loadKey<concreteprotocol::PackingKeyswitchKey, PackingKeyswitchKey>(
            (std::string)path));
    packingKeyswitchKeys.push_back(key);
  }

  ClientKeyset clientKeyset = ClientKeyset{secretKeys};
  ServerKeyset serverKeyset =
      ServerKeyset{bootstrapKeys, keyswitchKeys, packingKeyswitchKeys};
  Keyset keyset = Keyset{serverKeyset, clientKeyset};

  return keyset;
}

Result<void> saveKeys(Keyset &keyset, llvm::SmallString<0> &folderPath) {
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

  auto clientKeyset = keyset.client;
  auto serverKeyset = keyset.server;

  // Save LWE secret keys
  for (auto key : clientKeyset.lweSecretKeys) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(
        path, "secretKey_" + std::to_string(key.getInfo().asReader().getId()));
    OUTCOME_TRYV(saveKey<concreteprotocol::LweSecretKey, LweSecretKey>(
        key, path.c_str()));
  }
  // Save bootstrap keys
  for (auto key : serverKeyset.lweBootstrapKeys) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(
        path, "pbsKey_" + std::to_string(key.getInfo().asReader().getId()));
    OUTCOME_TRYV(saveKey<concreteprotocol::LweBootstrapKey, LweBootstrapKey>(
        key, path.c_str()));
  }
  // Save keyswitch keys
  for (auto key : serverKeyset.lweKeyswitchKeys) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(
        path, "ksKey_" + std::to_string(key.getInfo().asReader().getId()));
    OUTCOME_TRYV(saveKey<concreteprotocol::LweKeyswitchKey, LweKeyswitchKey>(
        key, path.c_str()));
  }
  // Save packing keyswitch keys
  for (auto key : serverKeyset.packingKeyswitchKeys) {
    llvm::SmallString<0> path = folderIncompletePath;
    llvm::sys::path::append(
        path, "pksKey_" + std::to_string(key.getInfo().asReader().getId()));
    OUTCOME_TRYV(
        saveKey<concreteprotocol::PackingKeyswitchKey, PackingKeyswitchKey>(
            key, path.c_str()));
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

KeysetCache::KeysetCache(std::string backingDirectoryPath) {
  // check key;
  this->backingDirectoryPath = backingDirectoryPath;
}

Result<Keyset>
KeysetCache::getKeyset(const Message<concreteprotocol::KeysetInfo> &keysetInfo,
                       __uint128_t secret_seed, __uint128_t encryption_seed) {
  std::string hashString = keysetInfo.asReader().toString().flatten().cStr() +
                           std::to_string((uint64_t)secret_seed) +
                           std::to_string((uint64_t)(secret_seed >> 64)) +
                           std::to_string((uint64_t)encryption_seed) +
                           std::to_string((uint64_t)(encryption_seed >> 64));

  size_t hash = std::hash<std::string>{}(hashString);
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  getApproval();
#endif

  llvm::SmallString<0> folderPath =
      llvm::SmallString<0>(this->backingDirectoryPath);
  llvm::sys::path::append(folderPath, std::to_string(hash));

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
    // Once it has been generated by another process (or was already here)
    auto keys = loadKeysFromFiles(keysetInfo, secret_seed, encryption_seed,
                                  std::string(folderPath));
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

  auto encryptionCsprng = csprng::EncryptionCSPRNG(encryption_seed);
  auto secretCsprng = csprng::SecretCSPRNG(secret_seed);
  Keyset keyset(keysetInfo, secretCsprng, encryptionCsprng);

  OUTCOME_TRYV(saveKeys(keyset, folderPath));

  return std::move(keyset);
}

} // namespace keysets
} // namespace concretelang

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Keysets.h"
#include "capnp/message.h"
#include "concrete-cpu.h"
#include "concrete-optimizer.hpp"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keys.h"
#include "concretelang/Common/Security.h"
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
  return fromProto(proto.asReader());
}

ClientKeyset
ClientKeyset::fromProto(concreteprotocol::ClientKeyset::Reader reader) {
  auto output = ClientKeyset();
  for (auto skProto : reader.getLweSecretKeys()) {
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
  return fromProto(proto.asReader());
}

ServerKeyset
ServerKeyset::fromProto(concreteprotocol::ServerKeyset::Reader reader) {
  auto output = ServerKeyset();
  for (auto bskProto : reader.getLweBootstrapKeys()) {
    output.lweBootstrapKeys.push_back(LweBootstrapKey::fromProto(bskProto));
  }

  for (auto kskProto : reader.getLweKeyswitchKeys()) {
    output.lweKeyswitchKeys.push_back(LweKeyswitchKey::fromProto(kskProto));
  }

  for (auto pkskProto : reader.getPackingKeyswitchKeys()) {
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
               SecretCSPRNG &secretCsprng, EncryptionCSPRNG &encryptionCsprng,
               std::map<uint32_t, LweSecretKey> lweSecretKeys) {
  for (auto keyInfo : info.asReader().getLweSecretKeys()) {
    if (lweSecretKeys.count(keyInfo.getId())) {
      // use provided key
      auto lweSk = lweSecretKeys.at(keyInfo.getId());
      assert(keyInfo.toString().flatten() ==
                 lweSk.getInfo().asReader().toString().flatten() &&
             "provided key info doesn't match expected ones");
      client.lweSecretKeys.push_back(lweSk);
    } else {
      // generate new key
      client.lweSecretKeys.push_back(LweSecretKey(
          (Message<concreteprotocol::LweSecretKeyInfo>)keyInfo, secretCsprng));
    }
  }
  for (auto keyInfo : info.asReader().getLweBootstrapKeys()) {
    server.lweBootstrapKeys.push_back(LweBootstrapKey(
        (Message<concreteprotocol::LweBootstrapKeyInfo>)keyInfo,
        client.lweSecretKeys[keyInfo.getInputId()],
        client.lweSecretKeys[keyInfo.getOutputId()], encryptionCsprng));
  }
  for (auto keyInfo : info.asReader().getLweKeyswitchKeys()) {
    server.lweKeyswitchKeys.push_back(LweKeyswitchKey(
        (Message<concreteprotocol::LweKeyswitchKeyInfo>)keyInfo,
        client.lweSecretKeys[keyInfo.getInputId()],
        client.lweSecretKeys[keyInfo.getOutputId()], encryptionCsprng));
  }
  for (auto keyInfo : info.asReader().getPackingKeyswitchKeys()) {
    server.packingKeyswitchKeys.push_back(PackingKeyswitchKey(
        (Message<concreteprotocol::PackingKeyswitchKeyInfo>)keyInfo,
        client.lweSecretKeys[keyInfo.getInputId()],
        client.lweSecretKeys[keyInfo.getOutputId()], encryptionCsprng));
  }
}

Keyset Keyset::fromProto(const Message<concreteprotocol::Keyset> &proto) {
  return fromProto(proto.asReader());
}

Keyset Keyset::fromProto(concreteprotocol::Keyset::Reader reader) {
  auto server = ServerKeyset::fromProto(reader.getServer());
  auto client = ClientKeyset::fromProto(reader.getClient());

  return {server, client};
}

Message<concreteprotocol::Keyset> Keyset::toProto() const {
  auto output = Message<concreteprotocol::Keyset>();
  // we inlined call to server.toProto() to avoid a single big copy of the
  // server keyset. With this, we only do copies of individual keys.
  auto serverKeyset = output.asBuilder().initServer();
  serverKeyset.initLweBootstrapKeys(server.lweBootstrapKeys.size());
  for (size_t i = 0; i < server.lweBootstrapKeys.size(); i++) {
    serverKeyset.getLweBootstrapKeys().setWithCaveats(
        i, server.lweBootstrapKeys[i].toProto().asReader());
  }

  serverKeyset.initLweKeyswitchKeys(server.lweKeyswitchKeys.size());
  for (size_t i = 0; i < server.lweKeyswitchKeys.size(); i++) {
    serverKeyset.getLweKeyswitchKeys().setWithCaveats(
        i, server.lweKeyswitchKeys[i].toProto().asReader());
  }

  serverKeyset.initPackingKeyswitchKeys(server.packingKeyswitchKeys.size());
  for (size_t i = 0; i < server.packingKeyswitchKeys.size(); i++) {
    serverKeyset.getPackingKeyswitchKeys().setWithCaveats(
        i, server.packingKeyswitchKeys[i].toProto().asReader());
  }
  // client serialization is not inlined as keys aren't that big
  auto clientProto = client.toProto();
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
                       __uint128_t secret_seed, __uint128_t encryption_seed,
                       std::map<uint32_t, LweSecretKey> lweSecretKeys) {
  std::string hashString = keysetInfo.asReader().toString().flatten().cStr() +
                           std::to_string((uint64_t)secret_seed) +
                           std::to_string((uint64_t)(secret_seed >> 64)) +
                           std::to_string((uint64_t)encryption_seed) +
                           std::to_string((uint64_t)(encryption_seed >> 64));

  // hash initial keys if any
  if (lweSecretKeys.size()) {
    hashString += "InitSKsSig:";
    for (auto sk : lweSecretKeys) {
      // key info
      hashString += sk.second.getInfo().asReader().toString().flatten().cStr();
      // key buffer
      std::vector<uint64_t> buffer = sk.second.getBuffer();
      size_t hash = std::hash<std::string_view>{}(
          {reinterpret_cast<const char *>(buffer.data()),
           buffer.size() * sizeof(uint64_t)});
      hashString += std::to_string(hash);
      hashString += ",";
    }
  }

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
  Keyset keyset(keysetInfo, secretCsprng, encryptionCsprng, lweSecretKeys);

  OUTCOME_TRYV(saveKeys(keyset, folderPath));

  return std::move(keyset);
}

Message<concreteprotocol::KeysetInfo>
generateKeysetInfoFromParameters(CircuitKeys parameters,
                                 concrete_optimizer::Options options) {
  auto output = Message<concreteprotocol::KeysetInfo>{};
  auto curve = ::concretelang::security::getSecurityCurve(
      options.security_level, ::concretelang::security::BINARY);

  auto skLen = (int)parameters.secret_keys.size();
  auto skBuilder = output.asBuilder().initLweSecretKeys(skLen);
  for (auto sk : llvm::enumerate(parameters.secret_keys)) {
    auto output = Message<concreteprotocol::LweSecretKeyInfo>();
    output.asBuilder().setId(sk.value().identifier);
    output.asBuilder().getParams().setIntegerPrecision(64);
    output.asBuilder().getParams().setLweDimension(sk.value().polynomial_size *
                                                   sk.value().glwe_dimension);
    output.asBuilder().getParams().setKeyType(
        ::concreteprotocol::KeyType::BINARY);
    skBuilder.setWithCaveats(sk.index(), output.asReader());
  }

  auto bskLen = (int)parameters.bootstrap_keys.size();
  auto bskBuilder = output.asBuilder().initLweBootstrapKeys(bskLen);
  for (auto bsk : llvm::enumerate(parameters.bootstrap_keys)) {
    auto output = Message<concreteprotocol::LweBootstrapKeyInfo>();
    output.asBuilder().setId(bsk.value().identifier);
    output.asBuilder().setInputId(bsk.value().input_key.identifier);
    output.asBuilder().setOutputId(bsk.value().output_key.identifier);
    output.asBuilder().getParams().setLevelCount(
        bsk.value().br_decomposition_parameter.level);
    output.asBuilder().getParams().setBaseLog(
        bsk.value().br_decomposition_parameter.log2_base);
    output.asBuilder().getParams().setGlweDimension(
        bsk.value().output_key.glwe_dimension);
    output.asBuilder().getParams().setPolynomialSize(
        bsk.value().output_key.polynomial_size);
    output.asBuilder().getParams().setInputLweDimension(
        bsk.value().input_key.polynomial_size);
    output.asBuilder().getParams().setIntegerPrecision(64);
    output.asBuilder().getParams().setKeyType(
        concreteprotocol::KeyType::BINARY);
    output.asBuilder().getParams().setVariance(
        curve->getVariance(bsk.value().output_key.glwe_dimension,
                           bsk.value().output_key.polynomial_size, 64));
    bskBuilder.setWithCaveats(bsk.index(), output.asReader());
  }

  auto kskLen = (int)parameters.keyswitch_keys.size();
  auto ckskLen = (int)parameters.conversion_keyswitch_keys.size();
  auto kskBuilder = output.asBuilder().initLweKeyswitchKeys(kskLen + ckskLen);
  for (auto ksk : llvm::enumerate(parameters.keyswitch_keys)) {
    auto output = Message<concreteprotocol::LweKeyswitchKeyInfo>();
    output.asBuilder().setId(ksk.value().identifier);
    output.asBuilder().setInputId(ksk.value().input_key.identifier);
    output.asBuilder().setOutputId(ksk.value().output_key.identifier);
    output.asBuilder().getParams().setLevelCount(
        ksk.value().ks_decomposition_parameter.level);
    output.asBuilder().getParams().setBaseLog(
        ksk.value().ks_decomposition_parameter.log2_base);
    output.asBuilder().getParams().setIntegerPrecision(64);
    output.asBuilder().getParams().setInputLweDimension(
        ksk.value().input_key.glwe_dimension *
        ksk.value().input_key.polynomial_size);
    output.asBuilder().getParams().setOutputLweDimension(
        ksk.value().output_key.glwe_dimension *
        ksk.value().output_key.polynomial_size);
    output.asBuilder().getParams().setKeyType(
        concreteprotocol::KeyType::BINARY);
    output.asBuilder().getParams().setVariance(
        curve->getVariance(1,
                           ksk.value().output_key.glwe_dimension *
                               ksk.value().output_key.polynomial_size,
                           64));
    kskBuilder.setWithCaveats(ksk.index(), output.asReader());
  }
  for (auto ksk : llvm::enumerate(parameters.conversion_keyswitch_keys)) {
    auto output = Message<concreteprotocol::LweKeyswitchKeyInfo>();
    output.asBuilder().setId(ksk.value().identifier);
    output.asBuilder().setInputId(ksk.value().input_key.identifier);
    output.asBuilder().setOutputId(ksk.value().output_key.identifier);
    output.asBuilder().getParams().setLevelCount(
        ksk.value().ks_decomposition_parameter.level);
    output.asBuilder().getParams().setBaseLog(
        ksk.value().ks_decomposition_parameter.log2_base);
    output.asBuilder().getParams().setIntegerPrecision(64);
    output.asBuilder().getParams().setInputLweDimension(
        ksk.value().input_key.glwe_dimension *
        ksk.value().input_key.polynomial_size);
    output.asBuilder().getParams().setOutputLweDimension(
        ksk.value().output_key.glwe_dimension *
        ksk.value().output_key.polynomial_size);
    output.asBuilder().getParams().setKeyType(
        concreteprotocol::KeyType::BINARY);
    output.asBuilder().getParams().setVariance(
        curve->getVariance(1,
                           ksk.value().output_key.glwe_dimension *
                               ksk.value().output_key.polynomial_size,
                           64));
    kskBuilder.setWithCaveats(ksk.index() + kskLen, output.asReader());
  }
  return output;
}

Message<concreteprotocol::KeysetInfo> keysetInfoFromVirtualCircuit(
    std::vector<concrete_optimizer::utils::PartitionDefinition> partitionDefs,
    bool generateFks, std::optional<concrete_optimizer::Options> options) {

  rust::Vec<concrete_optimizer::utils::PartitionDefinition> rustPartitionDefs{};
  for (auto def : partitionDefs) {
    rustPartitionDefs.push_back(def);
  }

  auto defaultOptions = concrete_optimizer::Options{};
  defaultOptions.security_level = 128;
  defaultOptions.maximum_acceptable_error_probability = 0.000063342483999973;
  defaultOptions.key_sharing = true;
  defaultOptions.ciphertext_modulus_log = 64;
  defaultOptions.fft_precision = 53;

  auto opts = options.value_or(defaultOptions);

  auto parameters = concrete_optimizer::utils::generate_virtual_keyset_info(
      rustPartitionDefs, generateFks, opts);

  return generateKeysetInfoFromParameters(parameters, opts);
}

} // namespace keysets
} // namespace concretelang

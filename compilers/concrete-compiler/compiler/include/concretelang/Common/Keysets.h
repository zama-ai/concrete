// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_KEYSETS_H
#define CONCRETELANG_COMMON_KEYSETS_H

#include "concrete-optimizer.hpp"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keys.h"
#include <functional>
#include <map>
#include <memory>
#include <stdlib.h>
#include <string>

using concretelang::error::Result;
using concretelang::error::StringError;
using concretelang::keys::LweBootstrapKey;
using concretelang::keys::LweKeyswitchKey;
using concretelang::keys::LweSecretKey;
using concretelang::keys::PackingKeyswitchKey;

namespace concretelang {
namespace keysets {

struct ClientKeyset {
  std::vector<LweSecretKey> lweSecretKeys;

  static ClientKeyset
  fromProto(const Message<concreteprotocol::ClientKeyset> &proto);

  static ClientKeyset fromProto(concreteprotocol::ClientKeyset::Reader reader);

  Message<concreteprotocol::ClientKeyset> toProto() const;
};

struct ServerKeyset {
  std::vector<LweBootstrapKey> lweBootstrapKeys;
  std::vector<LweKeyswitchKey> lweKeyswitchKeys;
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;

  static ServerKeyset
  fromProto(const Message<concreteprotocol::ServerKeyset> &proto);
  static ServerKeyset fromProto(concreteprotocol::ServerKeyset::Reader reader);

  Message<concreteprotocol::ServerKeyset> toProto() const;
};

struct Keyset {
  ServerKeyset server;
  ClientKeyset client;

  Keyset(){};

  /// @brief Generates a keyset from infos.
  ///
  /// This can be a fresh keyset if no key is specified in `lweSecretKeys`.
  /// Otherwise those keys are set first, then the rest of the key will be
  /// generated.
  ///
  /// @param info
  /// @param secretCsprng
  /// @param encryptionCsprng
  /// @param lweSecretKeys secret keys to initialize the keyset with
  Keyset(const Message<concreteprotocol::KeysetInfo> &info,
         concretelang::csprng::SecretCSPRNG &secretCsprng,
         csprng::EncryptionCSPRNG &encryptionCsprng,
         std::map<uint32_t, LweSecretKey> lweSecretKeys =
             std::map<uint32_t, LweSecretKey>());

  Keyset(ServerKeyset server, ClientKeyset client)
      : server(server), client(client) {}

  static Keyset fromProto(const Message<concreteprotocol::Keyset> &proto);
  static Keyset fromProto(concreteprotocol::Keyset::Reader reader);

  Message<concreteprotocol::Keyset> toProto() const;
};

class KeysetCache {
  std::string backingDirectoryPath;

public:
  KeysetCache(std::string backingDirectoryPath);

  Result<Keyset>
  getKeyset(const Message<concreteprotocol::KeysetInfo> &keysetInfo,
            __uint128_t secret_seed, __uint128_t encryption_seed,
            std::map<uint32_t, LweSecretKey> lweSecretKeys =
                std::map<uint32_t, LweSecretKey>());

private:
  KeysetCache() = default;
};

Message<concreteprotocol::KeysetInfo> keysetInfoFromVirtualCircuit(
    std::vector<concrete_optimizer::utils::PartitionDefinition> partitions,
    bool generate_fks, std::optional<concrete_optimizer::Options> options);

} // namespace keysets
} // namespace concretelang

#endif

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_KEYSETS_H
#define CONCRETELANG_COMMON_KEYSETS_H

#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keys.h"
#include <functional>
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

  Message<concreteprotocol::ClientKeyset> toProto() const;
};

struct ServerKeyset {
  std::vector<LweBootstrapKey> lweBootstrapKeys;
  std::vector<LweKeyswitchKey> lweKeyswitchKeys;
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;

  static ServerKeyset
  fromProto(const Message<concreteprotocol::ServerKeyset> &proto);

  Message<concreteprotocol::ServerKeyset> toProto() const;
};

struct Keyset {
  ServerKeyset server;
  ClientKeyset client;

  /// Generates a fresh keyset from infos.
  Keyset(const Message<concreteprotocol::KeysetInfo> &info, CSPRNG &csprng);

  Keyset(ServerKeyset server, ClientKeyset client)
      : server(server), client(client) {}

  static Keyset fromProto(const Message<concreteprotocol::Keyset> &proto);

  Message<concreteprotocol::Keyset> toProto() const;
};

class KeysetCache {
  std::string backingDirectoryPath;

public:
  KeysetCache(std::string backingDirectoryPath);

  Result<Keyset>
  getKeyset(const Message<concreteprotocol::KeysetInfo> &keysetInfo,
            uint64_t seed_msb, uint64_t seed_lsb);

private:
  KeysetCache() = default;
};

} // namespace keysets
} // namespace concretelang

#endif

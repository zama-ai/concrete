// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_KEYSETS_H
#define CONCRETELANG_COMMON_KEYSETS_H

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keys.h"
#include <functional>
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

  static ClientKeyset fromProto(const concreteprotocol::ClientKeyset &proto);
  concreteprotocol::ClientKeyset toProto();
};

struct ServerKeyset {
  std::vector<LweBootstrapKey> lweBootstrapKeys;
  std::vector<LweKeyswitchKey> lweKeyswitchKeys;
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;

  static ServerKeyset fromProto(const concreteprotocol::ServerKeyset &proto) ;
  concreteprotocol::ServerKeyset toProto() ;
};

struct Keyset {
  ServerKeyset server;
  ClientKeyset client;
  
  /// Generates a fresh keyset from infos.
  Keyset(const concreteprotocol::KeysetInfo &info,
	 csprng::SecretCSPRNG &secretCsprng,
	 csprng::EncryptionCSPRNG &encryptionCsprng);
  static Keyset fromProto(const concreteprotocol::Keyset &proto) ;
  concreteprotocol::Keyset toProto() ;

  Keyset(ServerKeyset server, ClientKeyset client): server(server), client(client){}
};

class KeysetCache {
  std::string backingDirectoryPath;

public:
  KeysetCache(std::string backingDirectoryPath);

  Result<Keyset> getKeyset(const concreteprotocol::KeysetInfo &keysetInfo,
	       	   uint64_t secret_seed_msb, uint64_t secret_seed_lsb,
       	           uint64_t encryption_seed_msb, uint64_t encryption_seed_lsb) ;

private:
  KeysetCache() = default;
};

} // namespace keysets
} // namespace concretelang

#endif

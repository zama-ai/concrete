// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_KEYS_H
#define CONCRETELANG_COMMON_KEYS_H

#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Protocol.h"
#include <memory>
#include <stdlib.h>
#include <vector>

using concretelang::csprng::CSPRNG;
using concretelang::protocol::Message;

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
namespace keys {

/// An object representing an lwe Secret key
class LweSecretKey {
  friend class Keyset;
  friend class KeysetCache;
  friend class LweBootstrapKey;
  friend class LweKeyswitchKey;
  friend class PackingKeyswitchKey;

public:
  LweSecretKey(Message<concreteprotocol::LweSecretKeyInfo> info,
               CSPRNG &csprng);
  LweSecretKey() = delete;
  LweSecretKey(std::shared_ptr<std::vector<uint64_t>> buffer,
               Message<concreteprotocol::LweSecretKeyInfo> info)
      : buffer(buffer), info(info){};

  static LweSecretKey
  fromProto(const Message<concreteprotocol::LweSecretKey> &proto);

  Message<concreteprotocol::LweSecretKey> toProto() const;

  const uint64_t *getRawPtr() const;

  size_t getSize() const;

  const Message<concreteprotocol::LweSecretKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer() const;

  typedef Message<concreteprotocol::LweSecretKeyInfo> InfoType;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  Message<concreteprotocol::LweSecretKeyInfo> info;
};

class LweBootstrapKey {
  friend class Keyset;

public:
  LweBootstrapKey(Message<concreteprotocol::LweBootstrapKeyInfo> info,
                  const LweSecretKey &inputKey, const LweSecretKey &outputKey,
                  CSPRNG &csprng);
  LweBootstrapKey() = delete;
  LweBootstrapKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                  Message<concreteprotocol::LweBootstrapKeyInfo> info)
      : buffer(buffer), info(info){};

  static LweBootstrapKey
  fromProto(const Message<concreteprotocol::LweBootstrapKey> &proto);

  Message<concreteprotocol::LweBootstrapKey> toProto() const;

  const uint64_t *getRawPtr() const;

  size_t getSize() const;

  const Message<concreteprotocol::LweBootstrapKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer() const;

  typedef Message<concreteprotocol::LweBootstrapKeyInfo> InfoType;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  Message<concreteprotocol::LweBootstrapKeyInfo> info;
};

class LweKeyswitchKey {
  friend class Keyset;

public:
  LweKeyswitchKey(Message<concreteprotocol::LweKeyswitchKeyInfo> info,
                  const LweSecretKey &inputKey, const LweSecretKey &outputKey,
                  CSPRNG &csprng);
  LweKeyswitchKey() = delete;
  LweKeyswitchKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                  Message<concreteprotocol::LweKeyswitchKeyInfo> info)
      : buffer(buffer), info(info){};

  static LweKeyswitchKey
  fromProto(const Message<concreteprotocol::LweKeyswitchKey> &proto);

  Message<concreteprotocol::LweKeyswitchKey> toProto() const;

  const uint64_t *getRawPtr() const;

  size_t getSize() const;

  const Message<concreteprotocol::LweKeyswitchKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer() const;

  typedef Message<concreteprotocol::LweKeyswitchKeyInfo> InfoType;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  Message<concreteprotocol::LweKeyswitchKeyInfo> info;
};

class PackingKeyswitchKey {
  friend class Keyset;

public:
  PackingKeyswitchKey(Message<concreteprotocol::PackingKeyswitchKeyInfo> info,
                      const LweSecretKey &inputKey,
                      const LweSecretKey &outputKey, CSPRNG &csprng);
  PackingKeyswitchKey() = delete;
  PackingKeyswitchKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                      Message<concreteprotocol::PackingKeyswitchKeyInfo> info)
      : buffer(buffer), info(info){};

  static PackingKeyswitchKey
  fromProto(const Message<concreteprotocol::PackingKeyswitchKey> &proto);

  Message<concreteprotocol::PackingKeyswitchKey> toProto() const;

  const uint64_t *getRawPtr() const;

  size_t getSize() const;

  const Message<concreteprotocol::PackingKeyswitchKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer() const;

  typedef Message<concreteprotocol::PackingKeyswitchKeyInfo> InfoType;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  Message<concreteprotocol::PackingKeyswitchKeyInfo> info;
};

} // namespace keys
} // namespace concretelang

#endif

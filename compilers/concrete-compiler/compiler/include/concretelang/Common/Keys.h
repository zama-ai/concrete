// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_KEYS_H
#define CONCRETELANG_COMMON_KEYS_H

#include <stdlib.h>
#include "concrete-protocol.pb.h"
#include "concretelang/Common/Csprng.h"

using concretelang::csprng::CSPRNG;

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
  LweSecretKey(const concreteprotocol::LweSecretKeyInfo &info, CSPRNG &csprng); 

  static LweSecretKey fromProto(const concreteprotocol::LweSecretKey &proto);

  concreteprotocol::LweSecretKey toProto();

  const uint64_t *getRawPtr() const;

  size_t getSize();

  const concreteprotocol::LweSecretKeyInfo& getInfo();

private:
  LweSecretKey() = default;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  concreteprotocol::LweSecretKeyInfo info;
};

class LweBootstrapKey {
  friend class Keyset;

public:
  LweBootstrapKey(const concreteprotocol::LweBootstrapKeyInfo &info,
                  const LweSecretKey &inputKey, const LweSecretKey &outputKey,
                  CSPRNG &csprng);

  static LweBootstrapKey fromProto(const concreteprotocol::LweBootstrapKey &proto);

  concreteprotocol::LweBootstrapKey toProto();  

  const uint64_t *getRawPtr() const;

  size_t getSize();

  const concreteprotocol::LweBootstrapKeyInfo& getInfo();
    
private:
  LweBootstrapKey() = default;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  concreteprotocol::LweBootstrapKeyInfo info;
};

class LweKeyswitchKey {
  friend class Keyset;

public:
  LweKeyswitchKey(const concreteprotocol::LweKeyswitchKeyInfo &info,
                  const LweSecretKey &inputKey, const LweSecretKey &outputKey,
                  CSPRNG &csprng);
    
  static LweKeyswitchKey fromProto(const concreteprotocol::LweKeyswitchKey &proto);
    
  concreteprotocol::LweKeyswitchKey toProto();
    
  const uint64_t *getRawPtr() const;
    
  size_t getSize();
    
  const concreteprotocol::LweKeyswitchKeyInfo& getInfo();

private:
  LweKeyswitchKey() = default;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  concreteprotocol::LweKeyswitchKeyInfo info;
};

class PackingKeyswitchKey {
  friend class Keyset;

public:
  PackingKeyswitchKey(const concreteprotocol::PackingKeyswitchKeyInfo &info,
                      const LweSecretKey &inputKey,
                      const LweSecretKey &outputKey, CSPRNG &csprng);
    
  static PackingKeyswitchKey
  fromProto(const concreteprotocol::PackingKeyswitchKey &proto);
    
  concreteprotocol::PackingKeyswitchKey toProto() ;
    
  const uint64_t *getRawPtr() const ;
    
  size_t getSize();
    
  const concreteprotocol::PackingKeyswitchKeyInfo& getInfo();

private:
  PackingKeyswitchKey() = default;

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  concreteprotocol::PackingKeyswitchKeyInfo info;
};

} // namespace keys
} // namespace concretelang

#endif

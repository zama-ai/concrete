// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_KEYS_H
#define CONCRETELANG_COMMON_KEYS_H

#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Protocol.h"
#include <memory>
#include <mutex>
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
  typedef Message<concreteprotocol::LweSecretKeyInfo> InfoType;

  LweSecretKey(Message<concreteprotocol::LweSecretKeyInfo> info,
               concretelang::csprng::SecretCSPRNG &csprng);
  LweSecretKey() = delete;
  LweSecretKey(std::shared_ptr<std::vector<uint64_t>> buffer,
               Message<concreteprotocol::LweSecretKeyInfo> info)
      : buffer(buffer), info(info){};

  static LweSecretKey
  fromProto(const Message<concreteprotocol::LweSecretKey> &proto);

  static LweSecretKey fromProto(concreteprotocol::LweSecretKey::Reader reader);

  Message<concreteprotocol::LweSecretKey> toProto() const;

  const uint64_t *getRawPtr() const;

  size_t getSize() const;

  const Message<concreteprotocol::LweSecretKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer() const;

  const std::vector<uint64_t> &getTransportBuffer() const {
    return getBuffer();
  };

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  Message<concreteprotocol::LweSecretKeyInfo> info;
};

class LweBootstrapKey {
public:
  typedef Message<concreteprotocol::LweBootstrapKeyInfo> InfoType;

  /// @brief Constructor of a bootstrap key that initialize according with the
  /// given specification.
  /// @param info The info of the key to initialize.
  /// @param inputKey The input secret key of the bootstraping key.
  /// @param outputKey The output secret key of the bootstraping key.
  /// @param csprng An encryption csprng that used to encrypt the secret keys.
  LweBootstrapKey(Message<concreteprotocol::LweBootstrapKeyInfo> info,
                  const LweSecretKey &inputKey, const LweSecretKey &outputKey,
                  concretelang::csprng::EncryptionCSPRNG &csprng);
  LweBootstrapKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                  Message<concreteprotocol::LweBootstrapKeyInfo> info)
      : seededBuffer(std::make_shared<std::vector<uint64_t>>()), buffer(buffer),
        info(info), decompress_mutext(std::make_shared<std::mutex>()),
        decompressed(std::make_shared<bool>(false)){};

  /// @brief Initialize the key from the protocol message.
  static LweBootstrapKey
  fromProto(const Message<concreteprotocol::LweBootstrapKey> &proto);

  /// @brief Initialize the key from a reader.
  static LweBootstrapKey
  fromProto(concreteprotocol::LweBootstrapKey::Reader reader);

  /// @brief Returns the serialized form of the key.
  Message<concreteprotocol::LweBootstrapKey> toProto() const;

  const Message<concreteprotocol::LweBootstrapKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer();

  const std::vector<uint64_t> &getTransportBuffer() const;

  void decompress();

private:
  LweBootstrapKey(Message<concreteprotocol::LweBootstrapKeyInfo> info)
      : seededBuffer(std::make_shared<std::vector<uint64_t>>()),
        buffer(std::make_shared<std::vector<uint64_t>>()), info(info),
        decompress_mutext(std::make_shared<std::mutex>()),
        decompressed(std::make_shared<bool>(false)){};
  LweBootstrapKey() = delete;

  /// @brief  The buffer of the seeded key if needed.
  std::shared_ptr<std::vector<uint64_t>> seededBuffer;

  /// @brief The buffer of the actual bootstrap key.
  std::shared_ptr<std::vector<uint64_t>> buffer;

  /// @brief The metadata of the bootrap key.
  Message<concreteprotocol::LweBootstrapKeyInfo> info;

  /// @brief Mutex to guard the decompression
  std::shared_ptr<std::mutex> decompress_mutext;

  /// @brief A boolean that indicates if the decompression is done or not
  std::shared_ptr<bool> decompressed;
};

class LweKeyswitchKey {
public:
  typedef Message<concreteprotocol::LweKeyswitchKeyInfo> InfoType;

  LweKeyswitchKey(Message<concreteprotocol::LweKeyswitchKeyInfo> info,
                  const LweSecretKey &inputKey, const LweSecretKey &outputKey,
                  concretelang::csprng::EncryptionCSPRNG &csprng);
  LweKeyswitchKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                  Message<concreteprotocol::LweKeyswitchKeyInfo> info)
      : seededBuffer(std::make_shared<std::vector<uint64_t>>()), buffer(buffer),
        info(info), decompress_mutext(std::make_shared<std::mutex>()),
        decompressed(std::make_shared<bool>(false)){};

  /// @brief Initialize the key from the protocol message.
  static LweKeyswitchKey
  fromProto(const Message<concreteprotocol::LweKeyswitchKey> &proto);

  /// @brief Initialize the key from a reader.
  static LweKeyswitchKey
  fromProto(concreteprotocol::LweKeyswitchKey::Reader reader);

  /// @brief Returns the serialized form of the key.
  Message<concreteprotocol::LweKeyswitchKey> toProto() const;

  const Message<concreteprotocol::LweKeyswitchKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer();

  const std::vector<uint64_t> &getTransportBuffer() const;

  void decompress();

private:
  LweKeyswitchKey(Message<concreteprotocol::LweKeyswitchKeyInfo> info)
      : seededBuffer(std::make_shared<std::vector<uint64_t>>()),
        buffer(std::make_shared<std::vector<uint64_t>>()), info(info),
        decompress_mutext(std::make_shared<std::mutex>()),
        decompressed(std::make_shared<bool>(false)){};

  /// @brief  The buffer of the seeded key if needed.
  std::shared_ptr<std::vector<uint64_t>> seededBuffer;

  /// @brief The buffer of the actual bootstrap key.
  std::shared_ptr<std::vector<uint64_t>> buffer;

  /// @brief The metadata of the bootrap key.
  Message<concreteprotocol::LweKeyswitchKeyInfo> info;

  /// @brief Mutex to guard the decompression
  std::shared_ptr<std::mutex> decompress_mutext;

  /// @brief A boolean that indicates if the decompression is done or not
  std::shared_ptr<bool> decompressed;
};

class PackingKeyswitchKey {
  friend class Keyset;

public:
  typedef Message<concreteprotocol::PackingKeyswitchKeyInfo> InfoType;

  PackingKeyswitchKey(Message<concreteprotocol::PackingKeyswitchKeyInfo> info,
                      const LweSecretKey &inputKey,
                      const LweSecretKey &outputKey,
                      concretelang::csprng::EncryptionCSPRNG &csprng);
  PackingKeyswitchKey() = delete;
  PackingKeyswitchKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                      Message<concreteprotocol::PackingKeyswitchKeyInfo> info)
      : buffer(buffer), info(info){};

  static PackingKeyswitchKey
  fromProto(const Message<concreteprotocol::PackingKeyswitchKey> &proto);

  static PackingKeyswitchKey
  fromProto(concreteprotocol::PackingKeyswitchKey::Reader reader);

  Message<concreteprotocol::PackingKeyswitchKey> toProto() const;

  const uint64_t *getRawPtr() const;

  size_t getSize() const;

  const Message<concreteprotocol::PackingKeyswitchKeyInfo> &getInfo() const;

  const std::vector<uint64_t> &getBuffer() const;

  const std::vector<uint64_t> &getTransportBuffer() const {
    return getBuffer();
  };

private:
  std::shared_ptr<std::vector<uint64_t>> buffer;
  Message<concreteprotocol::PackingKeyswitchKeyInfo> info;
};

} // namespace keys
} // namespace concretelang

#endif

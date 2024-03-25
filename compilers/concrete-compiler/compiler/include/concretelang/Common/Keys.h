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

template <typename ProtoKey, typename ProtoInfo, typename Key>
class KeyWithBuffer {
public:
  typedef Message<ProtoKey> Proto;

  /// @brief Initialize the key from the protocol message.
  static Key fromProto(const Proto &proto) {
    auto key = Key(Message<ProtoInfo>(proto.asReader().getInfo()));
    key.buffer = concretelang::protocol::protoPayloadToSharedVector<uint64_t>(
        proto.asReader().getPayload());
    return key;
  };

  /// @brief Returns the proto representation of the key.
  Proto toProto() const {
    Proto msg;
    auto builder = msg.asBuilder();
    builder.setInfo(getInfo().asReader());
    builder.setPayload(
        concretelang::protocol::vectorToProtoPayload(getTransportBuffer())
            .asReader());
    return std::move(msg);
  }

  /// @brief Returns info of the key.
  const Message<ProtoInfo> &getInfo() const { return this->info; }

  const std::vector<uint64_t> &getBufferConst() const { return *buffer; }

  /// @brief Returns the buffer which contains the raw key.
  virtual const std::vector<uint64_t> &getBuffer() { return *buffer; }

  /// @brief Returns the buffer used for the transport.
  virtual const std::vector<uint64_t> &getTransportBuffer() const {
    return *buffer;
  }

protected:
  virtual ~KeyWithBuffer(){};
  KeyWithBuffer(const Message<ProtoInfo> info)
      : info(info), buffer(std::make_shared<std::vector<uint64_t>>()){};

  /// @brief The metadata of the key.
  Message<ProtoInfo> info;

  /// @brief The buffer of the actual key.
  std::shared_ptr<std::vector<uint64_t>> buffer;
};

template <typename ProtoKey, typename ProtoInfo, typename Key>
class KeyWithCompression : public KeyWithBuffer<ProtoKey, ProtoInfo, Key> {

public:
  /// @brief Initialize the key from the protocol message.
  static Key fromProto(const Message<ProtoKey> &proto) {
    auto info = proto.asReader().getInfo();
    auto vector = concretelang::protocol::protoPayloadToSharedVector<uint64_t>(
        proto.asReader().getPayload());
    auto key = Key(Message<ProtoInfo>(info));
    if (info.getCompression() == concreteprotocol::Compression::NONE) {
      key.buffer = vector;
    } else {
      key.compressedBuffer = vector;
    }
    return key;
  };

  /// @brief Returns the buffer which contains the raw key, can lazily
  /// decompress the key if needed.
  const std::vector<uint64_t> &getBuffer() override {
    decompress();
    return *S::buffer;
  }

  /// @brief Returns the buffer used for the transport.
  const std::vector<uint64_t> &getTransportBuffer() const override {
    if (S::getInfo().asReader().getCompression() ==
        concreteprotocol::Compression::NONE) {
      return *S::buffer;
    } else {
      return *compressedBuffer;
    }
  }

  /// @brief Force the decompression of the key if not yet done.
  void decompress() {
    if (*decompressed)
      return;
    const std::lock_guard<std::mutex> guard(*decompress_mutext);
    if (*decompressed)
      return;
    this->_decompress();
    *decompressed = true;
  }

protected:
  virtual ~KeyWithCompression(){};
  KeyWithCompression(const Message<ProtoInfo> info)
      : KeyWithBuffer<ProtoKey, ProtoInfo, Key>(info),
        compressedBuffer(std::make_shared<std::vector<uint64_t>>()),
        decompress_mutext(std::make_shared<std::mutex>()),
        decompressed(std::make_shared<bool>(false)){};
  virtual void _decompress() = 0;

protected:
  /// @brief  The buffer of the compressed key if needed.
  std::shared_ptr<std::vector<uint64_t>> compressedBuffer;

  /// @brief Mutex to guard the decompression
  std::shared_ptr<std::mutex> decompress_mutext;

  /// @brief A boolean that indicates if the decompression is done or not
  std::shared_ptr<bool> decompressed;

private:
  typedef KeyWithBuffer<ProtoKey, ProtoInfo, Key> S;
};

class LweSecretKey
    : public KeyWithBuffer<concreteprotocol::LweSecretKey,
                           concreteprotocol::LweSecretKeyInfo, LweSecretKey> {

public:
  LweSecretKey(Message<concreteprotocol::LweSecretKeyInfo> info,
               concretelang::csprng::SecretCSPRNG &csprng);

  void encrypt(uint64_t *lwe_ciphertext_buffer, const uint64_t input,
               double variance, csprng::EncryptionCSPRNG &csprng) const;

  void decrypt(uint64_t &output, const uint64_t *lwe_ciphertext_buffer) const;

protected:
  friend class KeyWithBuffer;

  LweSecretKey(Message<concreteprotocol::LweSecretKeyInfo> info)
      : KeyWithBuffer(info){};
};

class LwePublicKey
    : public KeyWithBuffer<concreteprotocol::LwePublicKey,
                           concreteprotocol::LwePublicKeyInfo, LwePublicKey> {

public:
  LwePublicKey(Message<concreteprotocol::LwePublicKeyInfo> info,
               LweSecretKey &secretKey,
               concretelang::csprng::EncryptionCSPRNG &csprng);

  void encrypt(uint64_t *lwe_ciphertext_buffer, const uint64_t input,
               csprng::SecretCSPRNG &csprng) const;

protected:
  friend class KeyWithBuffer;

  LwePublicKey(Message<concreteprotocol::LwePublicKeyInfo> info)
      : KeyWithBuffer(info){};
};

class LweBootstrapKey
    : public KeyWithCompression<concreteprotocol::LweBootstrapKey,
                                concreteprotocol::LweBootstrapKeyInfo,
                                LweBootstrapKey> {
public:
  /// @brief Constructor of a bootstrap key that initialize according with the
  /// given specification.
  /// @param info The info of the key to initialize.
  /// @param inputKey The input secret key of the bootstraping key.
  /// @param outputKey The output secret key of the bootstraping key.
  /// @param csprng An encryption csprng that used to encrypt the secret keys.
  LweBootstrapKey(Message<concreteprotocol::LweBootstrapKeyInfo> info,
                  LweSecretKey &inputKey, LweSecretKey &outputKey,
                  concretelang::csprng::EncryptionCSPRNG &csprng);

protected:
  friend class KeyWithCompression;

  void _decompress() override;
  LweBootstrapKey(Message<concreteprotocol::LweBootstrapKeyInfo> info)
      : KeyWithCompression(info){};
};

class LweKeyswitchKey
    : public KeyWithCompression<concreteprotocol::LweKeyswitchKey,
                                concreteprotocol::LweKeyswitchKeyInfo,
                                LweKeyswitchKey> {
public:
  LweKeyswitchKey(Message<concreteprotocol::LweKeyswitchKeyInfo> info,
                  LweSecretKey &inputKey, LweSecretKey &outputKey,
                  concretelang::csprng::EncryptionCSPRNG &csprng);

protected:
  friend class KeyWithCompression;

  void _decompress() override;

  LweKeyswitchKey(Message<concreteprotocol::LweKeyswitchKeyInfo> info)
      : KeyWithCompression(info){};
};

class PackingKeyswitchKey
    : public KeyWithBuffer<concreteprotocol::PackingKeyswitchKey,
                           concreteprotocol::PackingKeyswitchKeyInfo,
                           PackingKeyswitchKey> {

public:
  PackingKeyswitchKey(Message<concreteprotocol::PackingKeyswitchKeyInfo> info,
                      LweSecretKey &inputKey, LweSecretKey &outputKey,
                      concretelang::csprng::EncryptionCSPRNG &csprng);

protected:
  friend class KeyWithBuffer;

  PackingKeyswitchKey(Message<concreteprotocol::PackingKeyswitchKeyInfo> info)
      : KeyWithBuffer(info){};
};

} // namespace keys
} // namespace concretelang

#endif

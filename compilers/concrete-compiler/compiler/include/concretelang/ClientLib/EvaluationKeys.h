// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_EVALUATION_KEYS_H_
#define CONCRETELANG_CLIENTLIB_EVALUATION_KEYS_H_

#include <cassert>
#include <memory>
#include <vector>

#include "concrete-cpu.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

void getRandomSeed(struct Uint128 *u128);

template <typename Csprng> class CSPRNG {
public:
  Csprng *ptr;

  CSPRNG() = delete;
  CSPRNG(CSPRNG &) = delete;

  CSPRNG(CSPRNG &&other) : ptr(other.ptr) {
    assert(ptr != nullptr);
    other.ptr = nullptr;
  };

  CSPRNG(Csprng *ptr) : ptr(ptr){};
};

class SoftCSPRNG : public CSPRNG<Csprng> {
public:
  SoftCSPRNG(__uint128_t seed);
  SoftCSPRNG() = delete;
  SoftCSPRNG(SoftCSPRNG &) = delete;
  SoftCSPRNG(SoftCSPRNG &&other);
  ~SoftCSPRNG();
};

class SecretCSPRNG : public CSPRNG<SecCsprng> {
public:
  SecretCSPRNG(__uint128_t seed);
  SecretCSPRNG() = delete;
  SecretCSPRNG(SecretCSPRNG &) = delete;
  SecretCSPRNG(SecretCSPRNG &&other);
  ~SecretCSPRNG();
};

class EncryptionCSPRNG : public CSPRNG<EncCsprng> {
public:
  EncryptionCSPRNG(__uint128_t seed);
  EncryptionCSPRNG() = delete;
  EncryptionCSPRNG(EncryptionCSPRNG &) = delete;
  EncryptionCSPRNG(EncryptionCSPRNG &&other);
  ~EncryptionCSPRNG();
};

/// @brief LweSecretKey implements tools for manipulating lwe secret key on
/// client.
class LweSecretKey {
  std::shared_ptr<std::vector<uint64_t>> _buffer;
  LweSecretKeyParam _parameters;

public:
  LweSecretKey() = delete;
  LweSecretKey(LweSecretKeyParam &parameters, SecretCSPRNG &csprng);
  LweSecretKey(std::shared_ptr<std::vector<uint64_t>> buffer,
               LweSecretKeyParam parameters)
      : _buffer(buffer), _parameters(parameters){};

  /// @brief Encrypt the plaintext to the lwe ciphertext buffer.
  void encrypt(uint64_t *ciphertext, uint64_t plaintext, double variance,
               EncryptionCSPRNG &csprng) const;

  /// @brief Decrypt the ciphertext to the plaintext
  void decrypt(const uint64_t *ciphertext, uint64_t &plaintext) const;

  /// @brief Returns the buffer that hold the keyswitch key.
  const uint64_t *buffer() const { return _buffer->data(); }
  size_t size() const { return _buffer->size(); }

  /// @brief Returns the parameters of the keyswicth key.
  LweSecretKeyParam parameters() const { return this->_parameters; }

  /// @brief Returns the lwe dimension of the secret key.
  size_t dimension() const { return parameters().dimension; }
};

/// @brief LweKeyswitchKey implements tools for manipulating keyswitch key on
/// client.
class LweKeyswitchKey {
private:
  std::shared_ptr<std::vector<uint64_t>> _buffer;
  KeyswitchKeyParam _parameters;

public:
  LweKeyswitchKey() = delete;
  LweKeyswitchKey(KeyswitchKeyParam &parameters, LweSecretKey &inputKey,
                  LweSecretKey &outputKey, EncryptionCSPRNG &csprng);
  LweKeyswitchKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                  KeyswitchKeyParam parameters)
      : _buffer(buffer), _parameters(parameters){};

  /// @brief Returns the buffer that hold the keyswitch key.
  const uint64_t *buffer() const { return _buffer->data(); }
  size_t size() const { return _buffer->size(); }

  /// @brief Returns the parameters of the keyswicth key.
  KeyswitchKeyParam parameters() const { return this->_parameters; }
};

/// @brief LweBootstrapKey implements tools for manipulating bootstrap key on
/// client.
class LweBootstrapKey {
private:
  std::shared_ptr<std::vector<uint64_t>> _buffer;
  BootstrapKeyParam _parameters;

public:
  LweBootstrapKey() = delete;
  LweBootstrapKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                  BootstrapKeyParam &parameters)
      : _buffer(buffer), _parameters(parameters){};
  LweBootstrapKey(BootstrapKeyParam &parameters, LweSecretKey &inputKey,
                  LweSecretKey &outputKey, EncryptionCSPRNG &csprng);

  ///// @brief Returns the buffer that hold the bootstrap key.
  const uint64_t *buffer() const { return _buffer->data(); }
  size_t size() const { return _buffer->size(); }

  /// @brief Returns the parameters of the bootsrap key.
  BootstrapKeyParam parameters() const { return this->_parameters; }
};

/// @brief PackingKeyswitchKey implements tools for manipulating privat packing
/// keyswitch key on client.
class PackingKeyswitchKey {
private:
  std::shared_ptr<std::vector<uint64_t>> _buffer;
  PackingKeyswitchKeyParam _parameters;

public:
  PackingKeyswitchKey() = delete;
  PackingKeyswitchKey(PackingKeyswitchKeyParam &parameters,
                      LweSecretKey &inputKey, LweSecretKey &outputKey,
                      EncryptionCSPRNG &csprng);
  PackingKeyswitchKey(std::shared_ptr<std::vector<uint64_t>> buffer,
                      PackingKeyswitchKeyParam parameters)
      : _buffer(buffer), _parameters(parameters){};

  /// @brief Returns the buffer that hold the keyswitch key.
  const uint64_t *buffer() const { return _buffer->data(); }
  size_t size() const { return _buffer->size(); }

  /// @brief Returns the parameters of the keyswicth key.
  PackingKeyswitchKeyParam parameters() const { return this->_parameters; }
};

// =============================================

/// Evalution keys required for execution.
class EvaluationKeys {
private:
  std::vector<LweKeyswitchKey> keyswitchKeys;
  std::vector<LweBootstrapKey> bootstrapKeys;
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;

public:
  EvaluationKeys() = delete;

  EvaluationKeys(const std::vector<LweKeyswitchKey> keyswitchKeys,
                 const std::vector<LweBootstrapKey> bootstrapKeys,
                 const std::vector<PackingKeyswitchKey> packingKeyswitchKeys)
      : keyswitchKeys(keyswitchKeys), bootstrapKeys(bootstrapKeys),
        packingKeyswitchKeys(packingKeyswitchKeys) {}

  const LweKeyswitchKey &getKeyswitchKey(size_t id) const {
    return this->keyswitchKeys[id];
  }
  const std::vector<LweKeyswitchKey> getKeyswitchKeys() const {
    return this->keyswitchKeys;
  }

  const LweBootstrapKey &getBootstrapKey(size_t id) const {
    return bootstrapKeys[id];
  }
  const std::vector<LweBootstrapKey> getBootstrapKeys() const {
    return this->bootstrapKeys;
  }

  const PackingKeyswitchKey &getPackingKeyswitchKey(size_t id) const {
    return this->packingKeyswitchKeys[id];
  };

  const std::vector<PackingKeyswitchKey> getPackingKeyswitchKeys() const {
    return this->packingKeyswitchKeys;
  }
};

// =============================================

} // namespace clientlib
} // namespace concretelang

#endif

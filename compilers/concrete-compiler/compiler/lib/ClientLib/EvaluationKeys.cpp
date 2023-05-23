// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/ClientParameters.h"

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
inline void getApproval() {
  std::cerr << "DANGER: You are generating an empty unsecure secret keys. "
               "Enter \"y\" to continue: ";
  char answer;
  std::cin >> answer;
  if (answer != 'y') {
    std::abort();
  }
}
#endif

namespace concretelang {
namespace clientlib {

void getRandomSeed(struct Uint128 *u128) {
  switch (concrete_cpu_crypto_secure_random_128(u128)) {
  case 1:
    break;
  case -1:
    llvm::errs() << "WARNING: The generated random seed is not crypto secure\n";
    break;
  default:
    assert(false && "Cannot instantiate a random seed");
  }
}

SoftCSPRNG::SoftCSPRNG(__uint128_t seed) : CSPRNG<Csprng>(nullptr) {
  ptr = (Csprng *)aligned_alloc(CSPRNG_ALIGN, CSPRNG_SIZE);
  struct Uint128 u128;
  if (seed == 0) {
    getRandomSeed(&u128);
  } else {
    for (int i = 0; i < 16; i++) {
      u128.little_endian_bytes[i] = seed >> (8 * i);
    }
  }
  concrete_cpu_construct_csprng(ptr, u128);
}

SoftCSPRNG::SoftCSPRNG(SoftCSPRNG &&other) : CSPRNG(other.ptr) {
  assert(ptr != nullptr);
  other.ptr = nullptr;
}

SoftCSPRNG::~SoftCSPRNG() {
  if (ptr != nullptr) {
    concrete_cpu_destroy_csprng(ptr);
    free(ptr);
  }
}

SecretCSPRNG::SecretCSPRNG(__uint128_t seed) : CSPRNG<SecCsprng>(nullptr) {
  ptr = (SecCsprng *)aligned_alloc(SECRET_CSPRNG_ALIGN, SECRET_CSPRNG_SIZE);
  struct Uint128 u128;
  if (seed == 0) {
    getRandomSeed(&u128);
  } else {
    for (int i = 0; i < 16; i++) {
      u128.little_endian_bytes[i] = seed >> (8 * i);
    }
  }
  concrete_cpu_construct_secret_csprng(ptr, u128);
}

SecretCSPRNG::SecretCSPRNG(SecretCSPRNG &&other) : CSPRNG(other.ptr) {
  assert(ptr != nullptr);
  other.ptr = nullptr;
}

SecretCSPRNG::~SecretCSPRNG() {
  if (ptr != nullptr) {
    concrete_cpu_destroy_secret_csprng(ptr);
    free(ptr);
  }
}

EncryptionCSPRNG::EncryptionCSPRNG(__uint128_t seed)
    : CSPRNG<EncCsprng>(nullptr) {
  ptr = (EncCsprng *)aligned_alloc(ENCRYPTION_CSPRNG_ALIGN,
                                   ENCRYPTION_CSPRNG_SIZE);
  struct Uint128 u128;
  if (seed == 0) {
    getRandomSeed(&u128);
  } else {
    for (int i = 0; i < 16; i++) {
      u128.little_endian_bytes[i] = seed >> (8 * i);
    }
  }
  concrete_cpu_construct_encryption_csprng(ptr, u128);
}

EncryptionCSPRNG::EncryptionCSPRNG(EncryptionCSPRNG &&other)
    : CSPRNG(other.ptr) {
  assert(ptr != nullptr);
  other.ptr = nullptr;
}

EncryptionCSPRNG::~EncryptionCSPRNG() {
  if (ptr != nullptr) {
    concrete_cpu_destroy_encryption_csprng(ptr);
    free(ptr);
  }
}

LweSecretKey::LweSecretKey(LweSecretKeyParam &parameters, SecretCSPRNG &csprng)
    : _parameters(parameters) {
  // Allocate the buffer
  _buffer = std::make_shared<std::vector<uint64_t>>();
  _buffer->resize(parameters.dimension);
#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  // In insecure debug mode, the secret key is filled with zeros.
  getApproval();
  for (uint64_t &val : *_buffer) {
    val = 0;
  }
#else
  // Initialize the lwe secret key buffer
  concrete_cpu_init_secret_key_u64(_buffer->data(), parameters.dimension,
                                   csprng.ptr);
#endif
}

void LweSecretKey::encrypt(uint64_t *ciphertext, uint64_t plaintext,
                           double variance, EncryptionCSPRNG &csprng) const {
  concrete_cpu_encrypt_lwe_ciphertext_u64(_buffer->data(), ciphertext,
                                          plaintext, parameters().dimension,
                                          variance, csprng.ptr);
}

void LweSecretKey::decrypt(const uint64_t *ciphertext,
                           uint64_t &plaintext) const {
  concrete_cpu_decrypt_lwe_ciphertext_u64(_buffer->data(), ciphertext,
                                          parameters().dimension, &plaintext);
}

LweKeyswitchKey::LweKeyswitchKey(KeyswitchKeyParam &parameters,
                                 LweSecretKey &inputKey,
                                 LweSecretKey &outputKey,
                                 EncryptionCSPRNG &csprng)
    : _parameters(parameters) {
  // Allocate the buffer
  size_t size;
  if (!_parameters.compression) {
    size = concrete_cpu_keyswitch_key_size_u64(
        _parameters.level, inputKey.dimension(), outputKey.dimension());
  } else {
    size = concrete_cpu_seeded_keyswitch_key_size_u64(_parameters.level,
                                                      inputKey.dimension());
  }
  _buffer = std::make_shared<std::vector<uint64_t>>();
  _buffer->resize(size);

  // Initialize the keyswitch key buffer
  if (!_parameters.compression) {
    concrete_cpu_init_lwe_keyswitch_key_u64(
        _buffer->data(), inputKey.buffer(), outputKey.buffer(),
        inputKey.dimension(), outputKey.dimension(), _parameters.level,
        _parameters.baseLog, _parameters.variance, csprng.ptr);
  } else {
    struct Uint128 u128;
    getRandomSeed(&u128);
    concrete_cpu_init_seeded_lwe_keyswitch_key_u64(
        _buffer->data(), inputKey.buffer(), outputKey.buffer(),
        inputKey.dimension(), outputKey.dimension(), _parameters.level,
        _parameters.baseLog, u128, _parameters.variance);
  }
}

LweBootstrapKey::LweBootstrapKey(BootstrapKeyParam &parameters,
                                 LweSecretKey &inputKey,
                                 LweSecretKey &outputKey,
                                 EncryptionCSPRNG &csprng)
    : _parameters(parameters) {
  // TODO
  size_t polynomial_size = outputKey.dimension() / _parameters.glweDimension;
  // Allocate the buffer
  size_t size;
  if (!_parameters.compression) {
    size = concrete_cpu_bootstrap_key_size_u64(
        _parameters.level, _parameters.glweDimension, polynomial_size,
        inputKey.dimension());
  } else {
    size = concrete_cpu_seeded_bootstrap_key_size_u64(
        _parameters.level, _parameters.glweDimension, polynomial_size,
        inputKey.dimension());
  }
  _buffer = std::make_shared<std::vector<uint64_t>>();
  _buffer->resize(size);

  // Initialize the keyswitch key buffer
  if (!_parameters.compression) {
    concrete_cpu_init_lwe_bootstrap_key_u64(
        _buffer->data(), inputKey.buffer(), outputKey.buffer(),
        inputKey.dimension(), polynomial_size, _parameters.glweDimension,
        _parameters.level, _parameters.baseLog, _parameters.variance,
        Parallelism::Rayon, csprng.ptr);
  } else {
    struct Uint128 u128;
    getRandomSeed(&u128);
    concrete_cpu_init_seeded_lwe_bootstrap_key_u64(
        _buffer->data(), inputKey.buffer(), outputKey.buffer(),
        inputKey.dimension(), polynomial_size, _parameters.glweDimension,
        _parameters.level, _parameters.baseLog, u128, _parameters.variance,
        Parallelism::Rayon);
  }
}

PackingKeyswitchKey::PackingKeyswitchKey(PackingKeyswitchKeyParam &params,
                                         LweSecretKey &inputKey,
                                         LweSecretKey &outputKey,
                                         EncryptionCSPRNG &csprng)
    : _parameters(params) {
  assert(_parameters.inputLweDimension == inputKey.dimension());
  assert(_parameters.glweDimension * _parameters.polynomialSize ==
         outputKey.dimension());

  // Allocate the buffer
  auto size = concrete_cpu_lwe_packing_keyswitch_key_size(
      _parameters.glweDimension, _parameters.polynomialSize, _parameters.level,
      _parameters.inputLweDimension);
  _buffer = std::make_shared<std::vector<uint64_t>>();
  _buffer->resize(size * (_parameters.glweDimension + 1));

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
      _buffer->data(), inputKey.buffer(), outputKey.buffer(),
      _parameters.inputLweDimension, _parameters.polynomialSize,
      _parameters.glweDimension, _parameters.level, _parameters.baseLog,
      _parameters.variance, Parallelism::Rayon, csprng.ptr);
}

} // namespace clientlib
} // namespace concretelang

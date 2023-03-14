// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concrete-cpu.h"
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

ConcreteCSPRNG::ConcreteCSPRNG(__uint128_t seed)
    : CSPRNG(nullptr, &CONCRETE_CSPRNG_VTABLE) {
  ptr = (Csprng *)aligned_alloc(CONCRETE_CSPRNG_ALIGN, CONCRETE_CSPRNG_SIZE);
  struct Uint128 u128;
  if (seed == 0) {
    switch (concrete_cpu_crypto_secure_random_128(&u128)) {
    case 1:
      break;
    case -1:
      llvm::errs()
          << "WARNING: The generated random seed is not crypto secure\n";
      break;
    default:
      assert(false && "Cannot instantiate a random seed");
    }

  } else {
    for (int i = 0; i < 16; i++) {
      u128.little_endian_bytes[i] = seed >> (8 * i);
    }
  }
  concrete_cpu_construct_concrete_csprng(ptr, u128);
}

ConcreteCSPRNG::ConcreteCSPRNG(ConcreteCSPRNG &&other)
    : CSPRNG(other.ptr, &CONCRETE_CSPRNG_VTABLE) {
  assert(ptr != nullptr);
  other.ptr = nullptr;
}

ConcreteCSPRNG::~ConcreteCSPRNG() {
  if (ptr != nullptr) {
    concrete_cpu_destroy_concrete_csprng(ptr);
    free(ptr);
  }
}

LweSecretKey::LweSecretKey(LweSecretKeyParam &parameters, CSPRNG &csprng)
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
                                   csprng.ptr, csprng.vtable);
#endif
}

void LweSecretKey::encrypt(uint64_t *ciphertext, uint64_t plaintext,
                           double variance, CSPRNG &csprng) const {
  concrete_cpu_encrypt_lwe_ciphertext_u64(_buffer->data(), ciphertext,
                                          plaintext, parameters().dimension,
                                          variance, csprng.ptr, csprng.vtable);
}

void LweSecretKey::decrypt(const uint64_t *ciphertext,
                           uint64_t &plaintext) const {
  concrete_cpu_decrypt_lwe_ciphertext_u64(_buffer->data(), ciphertext,
                                          parameters().dimension, &plaintext);
}

LweKeyswitchKey::LweKeyswitchKey(KeyswitchKeyParam &parameters,
                                 LweSecretKey &inputKey,
                                 LweSecretKey &outputKey, CSPRNG &csprng)
    : _parameters(parameters) {
  // Allocate the buffer
  auto size = concrete_cpu_keyswitch_key_size_u64(
      _parameters.level, _parameters.baseLog, inputKey.dimension(),
      outputKey.dimension());
  _buffer = std::make_shared<std::vector<uint64_t>>();
  _buffer->resize(size);

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_keyswitch_key_u64(
      _buffer->data(), inputKey.buffer(), outputKey.buffer(),
      inputKey.dimension(), outputKey.dimension(), _parameters.level,
      _parameters.baseLog, _parameters.variance, csprng.ptr, csprng.vtable);
}

LweBootstrapKey::LweBootstrapKey(BootstrapKeyParam &parameters,
                                 LweSecretKey &inputKey,
                                 LweSecretKey &outputKey, CSPRNG &csprng)
    : _parameters(parameters) {
  // TODO
  size_t polynomial_size = outputKey.dimension() / _parameters.glweDimension;
  // Allocate the buffer
  auto size = concrete_cpu_bootstrap_key_size_u64(
      _parameters.level, _parameters.glweDimension, polynomial_size,
      inputKey.dimension());
  _buffer = std::make_shared<std::vector<uint64_t>>();
  _buffer->resize(size);

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_bootstrap_key_u64(
      _buffer->data(), inputKey.buffer(), outputKey.buffer(),
      inputKey.dimension(), polynomial_size, _parameters.glweDimension,
      _parameters.level, _parameters.baseLog, _parameters.variance,
      Parallelism::Rayon, csprng.ptr, csprng.vtable);
}

PackingKeyswitchKey::PackingKeyswitchKey(PackingKeyswitchKeyParam &params,
                                         LweSecretKey &inputKey,
                                         LweSecretKey &outputKey,
                                         CSPRNG &csprng)
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
      _parameters.variance, Parallelism::Rayon, csprng.ptr, csprng.vtable);
}

} // namespace clientlib
} // namespace concretelang

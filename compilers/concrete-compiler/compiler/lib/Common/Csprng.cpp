// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cstddef>
#include <stdio.h>

#include "concrete-cpu.h"
#include "concretelang/Common/Csprng.h"
#include "llvm/Support/raw_ostream.h"

namespace concretelang {
namespace csprng {

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

} // namespace csprng
} // namespace concretelang

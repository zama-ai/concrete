// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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

void writeSeed(struct Uint128 seed, uint64_t *buffer) {
  buffer[0] = (uint64_t)seed.little_endian_bytes[0];
  buffer[0] += (uint64_t)seed.little_endian_bytes[1] << 8;
  buffer[0] += (uint64_t)seed.little_endian_bytes[2] << 16;
  buffer[0] += (uint64_t)seed.little_endian_bytes[3] << 24;
  buffer[0] += (uint64_t)seed.little_endian_bytes[4] << 32;
  buffer[0] += (uint64_t)seed.little_endian_bytes[5] << 40;
  buffer[0] += (uint64_t)seed.little_endian_bytes[6] << 48;
  buffer[0] += (uint64_t)seed.little_endian_bytes[7] << 56;
  buffer[1] = (uint64_t)seed.little_endian_bytes[8];
  buffer[1] += (uint64_t)seed.little_endian_bytes[9] << 8;
  buffer[1] += (uint64_t)seed.little_endian_bytes[10] << 16;
  buffer[1] += (uint64_t)seed.little_endian_bytes[11] << 24;
  buffer[1] += (uint64_t)seed.little_endian_bytes[12] << 32;
  buffer[1] += (uint64_t)seed.little_endian_bytes[13] << 40;
  buffer[1] += (uint64_t)seed.little_endian_bytes[14] << 48;
  buffer[1] += (uint64_t)seed.little_endian_bytes[15] << 56;
}

void readSeed(struct Uint128 &seed, uint64_t *buffer) {
  seed.little_endian_bytes[0] = buffer[0];
  seed.little_endian_bytes[1] = buffer[0] >> 8;
  seed.little_endian_bytes[2] = buffer[0] >> 16;
  seed.little_endian_bytes[3] = buffer[0] >> 24;
  seed.little_endian_bytes[4] = buffer[0] >> 32;
  seed.little_endian_bytes[5] = buffer[0] >> 40;
  seed.little_endian_bytes[6] = buffer[0] >> 48;
  seed.little_endian_bytes[7] = buffer[0] >> 56;
  seed.little_endian_bytes[8] = buffer[1];
  seed.little_endian_bytes[9] = buffer[1] >> 8;
  seed.little_endian_bytes[10] = buffer[1] >> 16;
  seed.little_endian_bytes[11] = buffer[1] >> 24;
  seed.little_endian_bytes[12] = buffer[1] >> 32;
  seed.little_endian_bytes[13] = buffer[1] >> 40;
  seed.little_endian_bytes[14] = buffer[1] >> 48;
  seed.little_endian_bytes[15] = buffer[1] >> 56;
}

} // namespace csprng
} // namespace concretelang

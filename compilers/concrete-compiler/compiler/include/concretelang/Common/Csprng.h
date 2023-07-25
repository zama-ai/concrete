// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_CSPRNG_H
#define CONCRETELANG_COMMON_CSPRNG_H

#include <stdlib.h>
#include <assert.h>
#include "concrete-cpu.h"

namespace concretelang {
namespace csprng {

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

} // namespace csprng
} // namespace concretelang

#endif

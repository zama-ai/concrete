// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cstddef>
#include <stdio.h>

#include "llvm/Support/raw_ostream.h"
#include "concrete-cpu.h"
#include "concretelang/Common/Csprng.h"

namespace concretelang {
namespace csprng {

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

} // namespace csprng
} // namespace concretelang

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_CSPRNG_H
#define CONCRETELANG_COMMON_CSPRNG_H

#include <assert.h>
#include <stdlib.h>

struct Csprng;
struct CsprngVtable;

namespace concretelang {
namespace csprng {

class CSPRNG {
public:
  struct Csprng *ptr;
  const struct CsprngVtable *vtable;

  CSPRNG() = delete;
  CSPRNG(CSPRNG &) = delete;

  CSPRNG(CSPRNG &&other) : ptr(other.ptr), vtable(other.vtable) {
    assert(ptr != nullptr);
    other.ptr = nullptr;
  };

  CSPRNG(Csprng *ptr, const CsprngVtable *vtable) : ptr(ptr), vtable(vtable){};
};

class ConcreteCSPRNG : public CSPRNG {
public:
  ConcreteCSPRNG(__uint128_t seed);
  ConcreteCSPRNG() = delete;
  ConcreteCSPRNG(ConcreteCSPRNG &) = delete;
  ConcreteCSPRNG(ConcreteCSPRNG &&other);
  ~ConcreteCSPRNG();
};

} // namespace csprng
} // namespace concretelang

#endif

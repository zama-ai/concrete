// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include <map>
#include <mutex>
#include <pthread.h>

extern "C" {
#include "concrete-ffi.h"
}

namespace mlir {
namespace concretelang {

typedef struct RuntimeContext {
  LweKeyswitchKey_u64 *ksk;
  LweBootstrapKey_u64 *bsk;
  std::map<pthread_t, Engine *> engines;
  std::mutex engines_map_guard;

  RuntimeContext() {}
  // Ensure that the engines map is not copied
  RuntimeContext(const RuntimeContext &ctx) : ksk(ctx.ksk), bsk(ctx.bsk) {}
  RuntimeContext(const RuntimeContext &&other)
      : ksk(other.ksk), bsk(other.bsk) {}
  ~RuntimeContext() {
    for (const auto &key : engines) {
      free_engine(key.second);
    }
  }

  RuntimeContext &operator=(const RuntimeContext &rhs) {
    ksk = rhs.ksk;
    bsk = rhs.bsk;
    return *this;
  }
} RuntimeContext;

} // namespace concretelang
} // namespace mlir

extern "C" {
LweKeyswitchKey_u64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context);

LweBootstrapKey_u64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context);

Engine *get_engine(mlir::concretelang::RuntimeContext *context);
}
#endif

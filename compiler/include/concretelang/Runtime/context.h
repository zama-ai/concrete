// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include <map>
#include <mutex>
#include <string>

extern "C" {
#include "concrete-ffi.h"
}

namespace mlir {
namespace concretelang {

typedef struct RuntimeContext {
  LweKeyswitchKey_u64 *ksk;
  LweBootstrapKey_u64 *bsk;
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
  std::map<std::string, Engine *> engines;
  std::mutex engines_map_guard;
#else
  Engine *engine;
#endif

  RuntimeContext()
#ifndef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
      : engine(nullptr)
#endif
  {
  }
  // Ensure that the engines map is not copied
  RuntimeContext(const RuntimeContext &ctx)
      : ksk(ctx.ksk), bsk(ctx.bsk)
#ifndef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
        ,
        engine(nullptr)
#endif
  {
  }
  RuntimeContext(const RuntimeContext &&other)
      : ksk(other.ksk), bsk(other.bsk)
#ifndef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
        ,
        engine(nullptr)
#endif
  {
  }
  ~RuntimeContext() {
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
    for (const auto &key : engines) {
      free_engine(key.second);
    }
#else
    if (engine != nullptr)
      free_engine(engine);
#endif
  }

  RuntimeContext &operator=(const RuntimeContext &rhs) {
    ksk = rhs.ksk;
    bsk = rhs.bsk;
#ifndef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
    engine = nullptr;
#endif
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

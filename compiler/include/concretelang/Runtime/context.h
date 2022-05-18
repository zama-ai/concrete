// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include <map>
#include <mutex>
#include <pthread.h>

#include "concretelang/ClientLib/EvaluationKeys.h"

extern "C" {
#include "concrete-ffi.h"
}

namespace mlir {
namespace concretelang {

typedef struct RuntimeContext {
  ::concretelang::clientlib::EvaluationKeys evaluationKeys;
  std::map<pthread_t, Engine *> engines;
  std::mutex engines_map_guard;

  RuntimeContext() {}

  // Ensure that the engines map is not copied
  RuntimeContext(const RuntimeContext &ctx)
      : evaluationKeys(ctx.evaluationKeys) {}
  RuntimeContext(const RuntimeContext &&other)
      : evaluationKeys(other.evaluationKeys) {}

  ~RuntimeContext() {
    for (const auto &key : engines) {
      free_engine(key.second);
    }
  }

  RuntimeContext &operator=(const RuntimeContext &rhs) {
    this->evaluationKeys = rhs.evaluationKeys;
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

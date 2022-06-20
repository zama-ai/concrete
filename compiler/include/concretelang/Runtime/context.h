// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include <assert.h>
#include <map>
#include <mutex>
#include <pthread.h>

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/Runtime/seeder.h"

#include "concrete-core-ffi.h"
#include "concretelang/Common/Error.h"

namespace mlir {
namespace concretelang {

typedef struct RuntimeContext {
  ::concretelang::clientlib::EvaluationKeys evaluationKeys;
  DefaultEngine *default_engine;
  std::map<pthread_t, FftwEngine *> fftw_engines;
  std::mutex engines_map_guard;

  RuntimeContext() {
    CAPI_ASSERT_ERROR(new_default_engine(best_seeder, &default_engine));
  }

  /// Ensure that the engines map is not copied
  RuntimeContext(const RuntimeContext &ctx)
      : evaluationKeys(ctx.evaluationKeys) {
    CAPI_ASSERT_ERROR(new_default_engine(best_seeder, &default_engine));
  }
  RuntimeContext(const RuntimeContext &&other)
      : evaluationKeys(other.evaluationKeys),
        default_engine(other.default_engine) {}

  ~RuntimeContext() {
    CAPI_ASSERT_ERROR(destroy_default_engine(default_engine));
    for (const auto &key : fftw_engines) {
      CAPI_ASSERT_ERROR(destroy_fftw_engine(key.second));
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
LweKeyswitchKey64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context);

FftwFourierLweBootstrapKey64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context);

DefaultEngine *get_engine(mlir::concretelang::RuntimeContext *context);

FftwEngine *get_fftw_engine(mlir::concretelang::RuntimeContext *context);
}
#endif

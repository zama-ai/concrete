// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Runtime/context.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/seeder.h"
#include <assert.h>
#include <stdio.h>

LweKeyswitchKey64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->evaluationKeys.getKsk();
}

FftwFourierLweBootstrapKey64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->evaluationKeys.getBsk();
}

DefaultEngine *get_engine(mlir::concretelang::RuntimeContext *context) {
  return context->default_engine;
}

FftwEngine *get_fftw_engine(mlir::concretelang::RuntimeContext *context) {
  pthread_t threadId = pthread_self();
  std::lock_guard<std::mutex> guard(context->engines_map_guard);
  auto engineIt = context->fftw_engines.find(threadId);
  if (engineIt == context->fftw_engines.end()) {
    FftwEngine *fftw_engine = nullptr;

    CAPI_ASSERT_ERROR(new_fftw_engine(&fftw_engine));

    engineIt =
        context->fftw_engines
            .insert(std::pair<pthread_t, FftwEngine *>(threadId, fftw_engine))
            .first;
  }
  assert(engineIt->second && "No engine available in context");
  return engineIt->second;
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <assert.h>
#include <concretelang/Runtime/context.h>
#include <stdio.h>

LweKeyswitchKey_u64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->evaluationKeys.getKsk();
}

LweBootstrapKey_u64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->evaluationKeys.getBsk();
}

// Instantiate one engine per thread on demand
Engine *get_engine(mlir::concretelang::RuntimeContext *context) {
  pthread_t threadId = pthread_self();
  std::lock_guard<std::mutex> guard(context->engines_map_guard);
  auto engineIt = context->engines.find(threadId);
  if (engineIt == context->engines.end()) {
    engineIt =
        context->engines
            .insert(std::pair<pthread_t, Engine *>(threadId, new_engine()))
            .first;
  }
  assert(engineIt->second && "No engine available in context");
  return engineIt->second;
}

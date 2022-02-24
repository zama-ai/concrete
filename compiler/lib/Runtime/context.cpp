// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <assert.h>
#include <concretelang/Runtime/context.h>
#include <stdio.h>

#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
#include <hpx/include/runtime.hpp>
#endif

LweKeyswitchKey_u64 *
get_keyswitch_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->ksk;
}

LweBootstrapKey_u64 *
get_bootstrap_key_u64(mlir::concretelang::RuntimeContext *context) {
  return context->bsk;
}

// Instantiate one engine per thread on demand
Engine *get_engine(mlir::concretelang::RuntimeContext *context) {
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
  std::string threadName = hpx::get_thread_name();
  std::lock_guard<std::mutex> guard(context->engines_map_guard);
  auto engineIt = context->engines.find(threadName);
  if (engineIt == context->engines.end()) {
    engineIt =
        context->engines
            .insert(std::pair<std::string, Engine *>(threadName, new_engine()))
            .first;
  }
  assert(engineIt->second && "No engine available in context");
  return engineIt->second;
#else
  return (context->engine == nullptr) ? context->engine = new_engine()
                                      : context->engine;
#endif
}

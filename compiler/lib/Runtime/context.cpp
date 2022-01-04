// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include <assert.h>
#include <concretelang/Runtime/context.h>
#include <stdio.h>

#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
#include <hpx/include/runtime.hpp>
#endif

LweKeyswitchKey_u64 *get_keyswitch_key(mlir::concretelang::RuntimeContext *context) {
  return context->ksk;
}

LweBootstrapKey_u64 *get_bootstrap_key(mlir::concretelang::RuntimeContext *context) {
  int err;
#ifdef CONCRETELANG_PARALLEL_EXECUTION_ENABLED
  std::string threadName = hpx::get_thread_name();
  auto bskIt = context->bsk.find(threadName);
  if (bskIt == context->bsk.end()) {
    bskIt = context->bsk
                .insert(std::pair<std::string, LweBootstrapKey_u64 *>(
                    threadName,
                    clone_lwe_bootstrap_key_u64(
                        &err, context->bsk["_concretelang_base_context_bsk"])))
                .first;
    if (err != 0)
      fprintf(stderr, "Runtime: cloning bootstrap key failed.\n");
  }
#else
  std::string baseName = "_concretelang_base_context_bsk";
  auto bskIt = context->bsk.find(baseName);
#endif
  assert(bskIt->second && "No bootstrap key available in context");
  return bskIt->second;
}

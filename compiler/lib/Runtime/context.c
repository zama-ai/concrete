#include "concrete-ffi.h"
#include "zamalang/Runtime/context.h"
#include <stdio.h>

LweKeyswitchKey_u64 *get_keyswitch_key(RuntimeContext *context) {
  return context->ksk;
}

LweBootstrapKey_u64 *get_bootstrap_key(RuntimeContext *context) {
  int err;
  LweBootstrapKey_u64 *clone =
    clone_lwe_bootstrap_key_u64(&err, context->bsk);
  if (err != 0)
    fprintf(stderr, "Runtime: cloning bootstrap key failed.\n");
  return clone;
}

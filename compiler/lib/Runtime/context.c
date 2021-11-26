#include "zamalang/Runtime/context.h"
#include <stdio.h>

LweKeyswitchKey_u64 *get_keyswitch_key(RuntimeContext *context) {
  return context->ksk;
}

LweBootstrapKey_u64 *get_bootstrap_key(RuntimeContext *context) {
  return context->bsk;
}

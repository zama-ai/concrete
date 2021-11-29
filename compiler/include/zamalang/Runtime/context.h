#ifndef ZAMALANG_RUNTIME_CONTEXT_H
#define ZAMALANG_RUNTIME_CONTEXT_H

#include "concrete-ffi.h"

typedef struct RuntimeContext {
  LweKeyswitchKey_u64 *ksk;
  LweBootstrapKey_u64 *bsk;
} RuntimeContext;

LweKeyswitchKey_u64 *get_keyswitch_key(RuntimeContext *context);

LweBootstrapKey_u64 *get_bootstrap_key(RuntimeContext *context);

#endif

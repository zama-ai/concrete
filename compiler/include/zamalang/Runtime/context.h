#ifndef ZAMALANG_RUNTIME_CONTEXT_H
#define ZAMALANG_RUNTIME_CONTEXT_H

#include "concrete-ffi.h"

typedef struct RuntimeContext {
  struct LweKeyswitchKey_u64 *ksk;
  struct LweBootstrapKey_u64 *bsk;
} RuntimeContext;

extern RuntimeContext *globalRuntimeContext;

RuntimeContext *createRuntimeContext(LweKeyswitchKey_u64 *ksk,
                                     LweBootstrapKey_u64 *bsk);

void setGlobalRuntimeContext(RuntimeContext *context);

RuntimeContext *getGlobalRuntimeContext();

LweKeyswitchKey_u64 *getGlobalKeyswitchKey();

LweBootstrapKey_u64 *getGlobalBootstrapKey();

LweKeyswitchKey_u64 *getKeyswitckKeyFromContext(RuntimeContext *context);

LweBootstrapKey_u64 *getBootstrapKeyFromContext(RuntimeContext *context);

bool checkError(int *err);

#endif

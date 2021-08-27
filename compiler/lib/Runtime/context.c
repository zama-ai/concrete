#include "zamalang/Runtime/context.h"
#include <stdio.h>

RuntimeContext *globalRuntimeContext;

RuntimeContext *createRuntimeContext(LweKeyswitchKey_u64 *ksk,
                                     LweBootstrapKey_u64 *bsk) {
  RuntimeContext *context = (RuntimeContext *)malloc(sizeof(RuntimeContext));
  context->ksk = ksk;
  context->bsk = bsk;
  return context;
}

void setGlobalRuntimeContext(RuntimeContext *context) {
  globalRuntimeContext = context;
}

RuntimeContext *getGlobalRuntimeContext() { return globalRuntimeContext; }

LweKeyswitchKey_u64 *getGlobalKeyswitchKey() {
  return globalRuntimeContext->ksk;
}

LweBootstrapKey_u64 *getGlobalBootstrapKey() {
  return globalRuntimeContext->bsk;
}

LweKeyswitchKey_u64 *getKeyswitckKeyFromContext(RuntimeContext *context) {
  return context->ksk;
}

LweBootstrapKey_u64 *getBootstrapKeyFromContext(RuntimeContext *context) {
  return context->bsk;
}

bool checkError(int *err) {
  switch (*err) {
  case ERR_INDEX_OUT_OF_BOUND:
    fprintf(stderr, "Runtime: index out of bound");
    break;
  case ERR_NULL_POINTER:
    fprintf(stderr, "Runtime: null pointer");
    break;
  case ERR_SIZE_MISMATCH:
    fprintf(stderr, "Runtime: size mismatch");
    break;
  default:
    return false;
  }
  return true;
}
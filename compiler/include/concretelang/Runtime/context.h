// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_RUNTIME_CONTEXT_H
#define CONCRETELANG_RUNTIME_CONTEXT_H

#include <map>
#include <string>

extern "C" {
#include "concrete-ffi.h"
}

namespace mlir {
namespace concretelang {
typedef struct RuntimeContext {
  LweKeyswitchKey_u64 *ksk;
  std::map<std::string, LweBootstrapKey_u64 *> bsk;

  ~RuntimeContext() {
    int err;
    for (const auto &key : bsk) {
    if (key.first != "_concretelang_base_context_bsk")
      free_lwe_bootstrap_key_u64(&err, key.second);
    }
  }
} RuntimeContext;
}
}

extern "C" {
LweKeyswitchKey_u64 *get_keyswitch_key(mlir::concretelang::RuntimeContext *context);

LweBootstrapKey_u64 *get_bootstrap_key(mlir::concretelang::RuntimeContext *context);
}
#endif

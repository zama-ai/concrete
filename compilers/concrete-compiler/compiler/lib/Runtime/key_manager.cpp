// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifdef CONCRETELANG_DATAFLOW_EXECUTION_ENABLED

#include "concretelang/Runtime/key_manager.hpp"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Runtime/context.h"

namespace mlir {
namespace concretelang {
namespace dfr {

RuntimeContextManager *_dfr_node_level_runtime_context_manager;

KeyWrapper<LweKeyswitchKey> getKsk(size_t keyId) {
  return KeyWrapper<LweKeyswitchKey>(std::vector<LweKeyswitchKey>{
      _dfr_node_level_runtime_context_manager->context->getKeys()
          .lweKeyswitchKeys[keyId]});
}

KeyWrapper<LweBootstrapKey> getBsk(size_t keyId) {
  return KeyWrapper<LweBootstrapKey>(std::vector<LweBootstrapKey>{
      _dfr_node_level_runtime_context_manager->context->getKeys()
          .lweBootstrapKeys[keyId]});
}

KeyWrapper<PackingKeyswitchKey> getPKsk(size_t keyId) {
  return KeyWrapper<PackingKeyswitchKey>(std::vector<PackingKeyswitchKey>{
      _dfr_node_level_runtime_context_manager->context->getKeys()
          .packingKeyswitchKeys[keyId]});
}

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif

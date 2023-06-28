// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_KEY_MANAGER_HPP
#define CONCRETELANG_DFR_KEY_MANAGER_HPP

#include <memory>
#include <mutex>
#include <stdlib.h>
#include <utility>

#include <hpx/include/runtime.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>

#include "compress_lwe/defines.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/Serializers.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/context.h"

#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/Common/Error.h"

namespace mlir {
namespace concretelang {
namespace dfr {

using namespace ::concretelang::clientlib;

struct RuntimeContextManager;
namespace {
static void *dl_handle;
static RuntimeContextManager *_dfr_node_level_runtime_context_manager;
} // namespace

template <typename LweKeyType, typename KeyParamType> struct KeyWrapper {
  std::vector<LweKeyType> keys;

  KeyWrapper() {}
  KeyWrapper(KeyWrapper &&moved) noexcept : keys(moved.keys) {}
  KeyWrapper(const KeyWrapper &kw) : keys(kw.keys) {}
  KeyWrapper &operator=(const KeyWrapper &rhs) {
    this->keys = rhs.keys;
    return *this;
  }
  KeyWrapper(std::vector<LweKeyType> keyvec) : keys(keyvec) {}
  friend class hpx::serialization::access;
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    ar << (size_t)keys.size();
    for (auto k : keys) {
      auto params = k.parameters();
      size_t param_size = sizeof(KeyParamType);
      ar << hpx::serialization::make_array((char *)&params, param_size);
      ar << (size_t)k.size();
      ar << hpx::serialization::make_array(k.buffer(), k.size());
    }
  }
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    size_t num_keys;
    ar >> num_keys;
    for (uint i = 0; i < num_keys; ++i) {
      KeyParamType params;
      size_t param_size = sizeof(params);
      ar >> hpx::serialization::make_array((char *)&params, param_size);
      size_t key_size;
      ar >> key_size;
      auto buffer = std::make_shared<std::vector<uint64_t>>();
      buffer->resize(key_size);
      ar >> hpx::serialization::make_array(buffer->data(), key_size);
      keys.push_back(LweKeyType(buffer, params));
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <typename LweKeyType, typename KeyParamType>
bool operator==(const KeyWrapper<LweKeyType, KeyParamType> &lhs,
                const KeyWrapper<LweKeyType, KeyParamType> &rhs) {
  if (lhs.keys.size() != rhs.keys.size())
    return false;
  for (size_t i = 0; i < lhs.keys.size(); ++i)
    if (lhs.keys[i].buffer() != rhs.keys[i].buffer())
      return false;
  return true;
}

/************************/
/* Context management.  */
/************************/

struct RuntimeContextManager {
  // TODO: this is only ok so long as we don't change keys. Once we
  // use multiple keys, should have a map.
  RuntimeContext *context;

  RuntimeContextManager() {
    context = nullptr;
    _dfr_node_level_runtime_context_manager = this;
  }

  void setContext(void *ctx) {
    assert(context == nullptr &&
           "Only one RuntimeContext can be used at a time.");

    // Root node broadcasts the evaluation keys and each remote
    // instantiates a local RuntimeContext.
    if (_dfr_is_root_node()) {
      RuntimeContext *context = (RuntimeContext *)ctx;

      KeyWrapper<LweKeyswitchKey, KeyswitchKeyParam> kskw(
          context->getKeys().getKeyswitchKeys());
      KeyWrapper<LweBootstrapKey, BootstrapKeyParam> bskw(
          context->getKeys().getBootstrapKeys());
      hpx::collectives::broadcast_to("ksk_keystore", kskw);
      hpx::collectives::broadcast_to("bsk_keystore", bskw);
    } else {
      auto kskFut = hpx::collectives::broadcast_from<
          KeyWrapper<LweKeyswitchKey, KeyswitchKeyParam>>("ksk_keystore");
      auto bskFut = hpx::collectives::broadcast_from<
          KeyWrapper<LweBootstrapKey, BootstrapKeyParam>>("bsk_keystore");
      KeyWrapper<LweKeyswitchKey, KeyswitchKeyParam> kskw = kskFut.get();
      KeyWrapper<LweBootstrapKey, BootstrapKeyParam> bskw = bskFut.get();

      std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
      std::optional<comp::CompressionKey> compkey;

      context = new mlir::concretelang::RuntimeContext(
          EvaluationKeys(kskw.keys, bskw.keys, packingKeyswitchKeys, compkey));
    }
  }

  RuntimeContext *getContext() { return context; }

  void clearContext() {
    if (context != nullptr)
      delete context;
    context = nullptr;
  }
};

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif

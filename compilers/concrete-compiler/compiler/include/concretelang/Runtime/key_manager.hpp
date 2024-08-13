// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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

#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/context.h"

#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keys.h"
#include "concretelang/Common/Keysets.h"

using concretelang::keys::LweBootstrapKey;
using concretelang::keys::LweKeyswitchKey;
using concretelang::keys::LweSecretKey;
using concretelang::keys::PackingKeyswitchKey;
using concretelang::keysets::ServerKeyset;

namespace mlir {
namespace concretelang {
namespace dfr {

struct RuntimeContextManager;
extern RuntimeContextManager *_dfr_node_level_runtime_context_manager;

template <typename LweKeyType> struct KeyWrapper {
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
      auto info = k.getInfo();
      auto maybe_info_string = info.writeBinaryToString();
      assert(maybe_info_string.has_value());
      auto info_string = maybe_info_string.value();
      ar << (size_t)info_string.size();
      ar << hpx::serialization::make_array(info_string.c_str(),
                                           info_string.size());
      ar << (size_t)k.getTransportBuffer().size();
      ar << hpx::serialization::make_array(k.getTransportBuffer().data(),
                                           k.getTransportBuffer().size());
    }
  }
  template <class Archive> void load(Archive &ar, const unsigned int version) {
    size_t num_keys;
    ar >> num_keys;
    for (uint i = 0; i < num_keys; ++i) {
      size_t info_size;
      ar >> info_size;
      auto info_vec = std::make_shared<std::vector<char>>();
      info_vec->resize(info_size);
      ar >> hpx::serialization::make_array(info_vec->data(), info_size);
      std::string info_string(info_vec->data(), info_size);
      typename LweKeyType::InfoType info;
      assert(info.readBinaryFromString(info_string).has_value());
      size_t key_size;
      ar >> key_size;
      auto buffer = std::make_shared<std::vector<uint64_t>>();
      buffer->resize(key_size);
      ar >> hpx::serialization::make_array(buffer->data(), key_size);
      keys.push_back(LweKeyType(buffer, info));
    }
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <typename LweKeyType>
bool operator==(const KeyWrapper<LweKeyType> &lhs,

                const KeyWrapper<LweKeyType> &rhs) {
  if (lhs.keys.size() != rhs.keys.size())
    return false;
  for (size_t i = 0; i < lhs.keys.size(); ++i)
    if (lhs.keys[i].getTransportBuffer() != rhs.keys[i].getTransportBuffer())
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
  bool allocated = false;
  bool lazy_key_transfer = false;

  RuntimeContextManager(bool lazy = false)
      : context(nullptr), lazy_key_transfer(lazy) {}

  void setContext(void *ctx) {
    assert(context == nullptr &&
           "Only one RuntimeContext can be used at a time.");
    context = (RuntimeContext *)ctx;

    if (lazy_key_transfer) {
      if (!_dfr_is_root_node()) {
        context =
            new mlir::concretelang::DistributedRuntimeContext(ServerKeyset());
        allocated = true;
      }
      return;
    }

    // When the root node does not require a context, we still need to
    // broadcast an empty keyset to remote nodes as they cannot know
    // ahead of time and avoid waiting for the broadcast. Instantiate
    // an empty context for this.
    if (_dfr_is_root_node() && ctx == nullptr) {
      context = new mlir::concretelang::RuntimeContext(ServerKeyset());
      allocated = true;
    }

    // Root node broadcasts the evaluation keys and each remote
    // instantiates a local RuntimeContext.
    if (_dfr_is_root_node()) {
      KeyWrapper<LweKeyswitchKey> kskw(context->getKeys().lweKeyswitchKeys);
      KeyWrapper<LweBootstrapKey> bskw(context->getKeys().lweBootstrapKeys);
      KeyWrapper<PackingKeyswitchKey> pkskw(
          context->getKeys().packingKeyswitchKeys);
      hpx::collectives::broadcast_to("ksk_keystore", kskw);
      hpx::collectives::broadcast_to("bsk_keystore", bskw);
      hpx::collectives::broadcast_to("pksk_keystore", pkskw);
    } else {
      auto kskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweKeyswitchKey>>(
              "ksk_keystore");
      auto bskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweBootstrapKey>>(
              "bsk_keystore");
      auto pkskFut =
          hpx::collectives::broadcast_from<KeyWrapper<PackingKeyswitchKey>>(
              "pksk_keystore");
      KeyWrapper<LweKeyswitchKey> kskw = kskFut.get();
      KeyWrapper<LweBootstrapKey> bskw = bskFut.get();
      KeyWrapper<PackingKeyswitchKey> pkskw = pkskFut.get();
      context = new mlir::concretelang::RuntimeContext(
          ServerKeyset{bskw.keys, kskw.keys, pkskw.keys});
    }
  }

  RuntimeContext *getContext() { return context; }

  void clearContext() {
    if (context != nullptr)
      // On root node deallocate only if allocated independently here
      if (!_dfr_is_root_node() || allocated)
        delete context;
    context = nullptr;
  }
};

KeyWrapper<LweKeyswitchKey> getKsk(size_t keyId);
KeyWrapper<LweBootstrapKey> getBsk(size_t keyId);
KeyWrapper<PackingKeyswitchKey> getPKsk(size_t keyId);

HPX_DEFINE_PLAIN_ACTION(getKsk, _get_ksk_action);
HPX_DEFINE_PLAIN_ACTION(getBsk, _get_bsk_action);
HPX_DEFINE_PLAIN_ACTION(getPKsk, _get_pksk_action);

} // namespace dfr
} // namespace concretelang
} // namespace mlir

#endif

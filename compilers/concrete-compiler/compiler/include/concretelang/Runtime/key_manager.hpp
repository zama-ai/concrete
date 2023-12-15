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
namespace {
static void *dl_handle;
static RuntimeContextManager *_dfr_node_level_runtime_context_manager;
} // namespace

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
      std::string info_string;
      ar >> info_string;
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

      KeyWrapper<LweKeyswitchKey> kskw(context->getKeys().lweKeyswitchKeys);
      KeyWrapper<LweBootstrapKey> bskw(context->getKeys().lweBootstrapKeys);
      hpx::collectives::broadcast_to("ksk_keystore", kskw);
      hpx::collectives::broadcast_to("bsk_keystore", bskw);
    } else {
      auto kskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweKeyswitchKey>>(
              "ksk_keystore");
      auto bskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweBootstrapKey>>(
              "bsk_keystore");
      KeyWrapper<LweKeyswitchKey> kskw = kskFut.get();
      KeyWrapper<LweBootstrapKey> bskw = bskFut.get();
      context = new mlir::concretelang::RuntimeContext(
          ServerKeyset{bskw.keys, kskw.keys, {}});
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

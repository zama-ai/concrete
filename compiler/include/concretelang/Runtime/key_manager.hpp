// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_KEY_MANAGER_HPP
#define CONCRETELANG_DFR_KEY_MANAGER_HPP

#include <memory>
#include <mutex>
#include <utility>

#include <hpx/include/runtime.hpp>
#include <hpx/modules/collectives.hpp>
#include <hpx/modules/serialization.hpp>

#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/context.h"

extern "C" {
#include "concrete-ffi.h"
}

namespace mlir {
namespace concretelang {
namespace dfr {

template <typename T> struct KeyManager;
struct RuntimeContextManager;
namespace {
static void *dl_handle;
static RuntimeContextManager *_dfr_node_level_runtime_context_manager;
} // namespace

template <typename LweKeyType> struct KeyWrapper {
  LweKeyType *key;

  KeyWrapper() : key(nullptr) {}
  KeyWrapper(LweKeyType *key) : key(key) {}
  KeyWrapper(KeyWrapper &&moved) noexcept : key(moved.key) {}
  KeyWrapper(const KeyWrapper &kw) : key(kw.key) {}
  KeyWrapper &operator=(const KeyWrapper &rhs) {
    this->key = rhs.key;
    return *this;
  }
  friend class hpx::serialization::access;
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const;
  template <class Archive> void load(Archive &ar, const unsigned int version);
  HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <typename LweKeyType>
bool operator==(const KeyWrapper<LweKeyType> &lhs,
                const KeyWrapper<LweKeyType> &rhs) {
  return lhs.key == rhs.key;
}

template <>
template <class Archive>
void KeyWrapper<LweBootstrapKey_u64>::save(Archive &ar,
                                           const unsigned int version) const {
  Buffer buffer = serialize_lwe_bootstrap_key_u64(key);
  ar << buffer.length;
  ar << hpx::serialization::make_array(buffer.pointer, buffer.length);
}
template <>
template <class Archive>
void KeyWrapper<LweBootstrapKey_u64>::load(Archive &ar,
                                           const unsigned int version) {
  size_t length;
  ar >> length;
  uint8_t *pointer = new uint8_t[length];
  ar >> hpx::serialization::make_array(pointer, length);
  BufferView buffer = {(const uint8_t *)pointer, length};
  key = deserialize_lwe_bootstrap_key_u64(buffer);
}

template <>
template <class Archive>
void KeyWrapper<LweKeyswitchKey_u64>::save(Archive &ar,
                                           const unsigned int version) const {
  Buffer buffer = serialize_lwe_keyswitching_key_u64(key);
  ar << buffer.length;
  ar << hpx::serialization::make_array(buffer.pointer, buffer.length);
}
template <>
template <class Archive>
void KeyWrapper<LweKeyswitchKey_u64>::load(Archive &ar,
                                           const unsigned int version) {
  size_t length;
  ar >> length;
  uint8_t *pointer = new uint8_t[length];
  ar >> hpx::serialization::make_array(pointer, length);
  BufferView buffer = {(const uint8_t *)pointer, length};
  key = deserialize_lwe_keyswitching_key_u64(buffer);
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
      LweKeyswitchKey_u64 *ksk = get_keyswitch_key_u64(context);
      LweBootstrapKey_u64 *bsk = get_bootstrap_key_u64(context);

      auto kskFut = hpx::collectives::broadcast_to(
          "ksk_keystore", KeyWrapper<LweKeyswitchKey_u64>(ksk));
      auto bskFut = hpx::collectives::broadcast_to(
          "bsk_keystore", KeyWrapper<LweBootstrapKey_u64>(bsk));
    } else {
      auto kskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweKeyswitchKey_u64>>(
              "ksk_keystore");
      auto bskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweBootstrapKey_u64>>(
              "bsk_keystore");

      context = new mlir::concretelang::RuntimeContext();
      context->evaluationKeys = ::concretelang::clientlib::EvaluationKeys(
          std::shared_ptr<::concretelang::clientlib::LweKeyswitchKey>(
              new ::concretelang::clientlib::LweKeyswitchKey(kskFut.get().key)),
          std::shared_ptr<::concretelang::clientlib::LweBootstrapKey>(
              new ::concretelang::clientlib::LweBootstrapKey(
                  bskFut.get().key)));
    }
  }

  RuntimeContext **getContext() { return &context; }

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

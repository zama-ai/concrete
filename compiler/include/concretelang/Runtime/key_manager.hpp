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

#include "concrete-core-ffi.h"
#include "concretelang/Common/Error.h"

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
  Buffer buffer;

  KeyWrapper() : key(nullptr) {}
  KeyWrapper(KeyWrapper &&moved) noexcept
      : key(moved.key), buffer(moved.buffer) {}
  KeyWrapper(LweKeyType *key);
  KeyWrapper(const KeyWrapper &kw) : key(kw.key), buffer(kw.buffer) {}
  KeyWrapper &operator=(const KeyWrapper &rhs) {
    this->key = rhs.key;
    this->buffer = rhs.buffer;
    return *this;
  }
  friend class hpx::serialization::access;
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const;
  template <class Archive> void load(Archive &ar, const unsigned int version);
  HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <>
KeyWrapper<LweKeyswitchKey64>::KeyWrapper(LweKeyswitchKey64 *key) : key(key) {

  DefaultSerializationEngine *engine;

  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));
  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(
      default_serialization_engine_serialize_lwe_keyswitch_key_u64(engine, key,
                                                                   &buffer));
}
template <>
KeyWrapper<LweBootstrapKey64>::KeyWrapper(LweBootstrapKey64 *key) : key(key) {

  DefaultSerializationEngine *engine;

  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(
      default_serialization_engine_serialize_lwe_bootstrap_key_u64(engine, key,
                                                                   &buffer));
}

template <typename LweKeyType>
bool operator==(const KeyWrapper<LweKeyType> &lhs,
                const KeyWrapper<LweKeyType> &rhs) {
  return lhs.key == rhs.key;
}

template <>
template <class Archive>
void KeyWrapper<LweBootstrapKey64>::save(Archive &ar,
                                         const unsigned int version) const {
  ar << buffer.length;
  ar << hpx::serialization::make_array(buffer.pointer, buffer.length);
}
template <>
template <class Archive>
void KeyWrapper<LweBootstrapKey64>::load(Archive &ar,
                                         const unsigned int version) {
  DefaultSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  ar >> buffer.length;
  buffer.pointer = new uint8_t[buffer.length];
  ar >> hpx::serialization::make_array(buffer.pointer, buffer.length);
  CAPI_ASSERT_ERROR(
      default_serialization_engine_deserialize_lwe_bootstrap_key_u64(
          engine, {buffer.pointer, buffer.length}, &key));
}

template <>
template <class Archive>
void KeyWrapper<LweKeyswitchKey64>::save(Archive &ar,
                                         const unsigned int version) const {
  ar << buffer.length;
  ar << hpx::serialization::make_array(buffer.pointer, buffer.length);
}
template <>
template <class Archive>
void KeyWrapper<LweKeyswitchKey64>::load(Archive &ar,
                                         const unsigned int version) {
  DefaultSerializationEngine *engine;

  // No Freeing as it doesn't allocate anything.
  CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

  ar >> buffer.length;
  buffer.pointer = new uint8_t[buffer.length];
  ar >> hpx::serialization::make_array(buffer.pointer, buffer.length);
  CAPI_ASSERT_ERROR(
      default_serialization_engine_deserialize_lwe_keyswitch_key_u64(
          engine, {buffer.pointer, buffer.length}, &key));
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
      LweKeyswitchKey64 *ksk = get_keyswitch_key_u64(context);
      LweBootstrapKey64 *bsk = get_bootstrap_key_u64(context);

      KeyWrapper<LweKeyswitchKey64> kskw(ksk);
      KeyWrapper<LweBootstrapKey64> bskw(bsk);
      hpx::collectives::broadcast_to("ksk_keystore", kskw);
      hpx::collectives::broadcast_to("bsk_keystore", bskw);
    } else {
      auto kskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweKeyswitchKey64>>(
              "ksk_keystore");
      auto bskFut =
          hpx::collectives::broadcast_from<KeyWrapper<LweBootstrapKey64>>(
              "bsk_keystore");

      KeyWrapper<LweKeyswitchKey64> kskw = kskFut.get();
      KeyWrapper<LweBootstrapKey64> bskw = bskFut.get();
      context = new mlir::concretelang::RuntimeContext();
      context->evaluationKeys = ::concretelang::clientlib::EvaluationKeys(
          std::shared_ptr<::concretelang::clientlib::LweKeyswitchKey>(
              new ::concretelang::clientlib::LweKeyswitchKey(kskw.key)),
          std::shared_ptr<::concretelang::clientlib::LweBootstrapKey>(
              new ::concretelang::clientlib::LweBootstrapKey(bskw.key)));
      delete (kskw.buffer.pointer);
      delete (bskw.buffer.pointer);
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

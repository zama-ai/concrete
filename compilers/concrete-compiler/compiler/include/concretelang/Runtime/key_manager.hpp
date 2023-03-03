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

#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/Runtime/DFRuntime.hpp"
#include "concretelang/Runtime/context.h"

#include "concretelang/Common/Error.h"

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
  // template <class Archive>
  // void save(Archive &ar, const unsigned int version) const;
  template <class Archive>
  void serialize(Archive &ar, const unsigned int version) const {}
  // template <class Archive> void load(Archive &ar, const unsigned int
  // version); HPX_SERIALIZATION_SPLIT_MEMBER()
};

template <typename LweKeyType>
bool operator==(const KeyWrapper<LweKeyType> &lhs,
                const KeyWrapper<LweKeyType> &rhs) {
  if (lhs.keys.size() != rhs.keys.size())
    return false;
  for (size_t i = 0; i < lhs.keys.size(); ++i)
    if (lhs.keys[i].buffer() != rhs.keys[i].buffer())
      return false;
  return true;
}

// template <>
// template <class Archive>
// void KeyWrapper<LweBootstrapKey>::save(Archive &ar,
//                                          const unsigned int version) const {
//   ar << buffer.length;
//   ar << hpx::serialization::make_array(buffer.pointer, buffer.length);
// }
// template <>
// template <class Archive>
// void KeyWrapper<LweBootstrapKey>::load(Archive &ar,
//                                          const unsigned int version) {
//   DefaultSerializationEngine *engine;

//   // No Freeing as it doesn't allocate anything.
//   CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

//   ar >> buffer.length;
//   buffer.pointer = new uint8_t[buffer.length];
//   ar >> hpx::serialization::make_array(buffer.pointer, buffer.length);
//   CAPI_ASSERT_ERROR(
//       default_serialization_engine_deserialize_lwe_bootstrap_key_u64(
//           engine, {buffer.pointer, buffer.length}, &key));
// }

// template <>
// template <class Archive>
// void KeyWrapper<LweKeyswitchKey>::save(Archive &ar,
//                                          const unsigned int version) const {
//   ar << buffer.length;
//   ar << hpx::serialization::make_array(buffer.pointer, buffer.length);
// }
// template <>
// template <class Archive>
// void KeyWrapper<LweKeyswitchKey>::load(Archive &ar,
//                                          const unsigned int version) {
//   DefaultSerializationEngine *engine;

//   // No Freeing as it doesn't allocate anything.
//   CAPI_ASSERT_ERROR(new_default_serialization_engine(&engine));

//   ar >> buffer.length;
//   buffer.pointer = new uint8_t[buffer.length];
//   ar >> hpx::serialization::make_array(buffer.pointer, buffer.length);
//   CAPI_ASSERT_ERROR(
//       default_serialization_engine_deserialize_lwe_keyswitch_key_u64(
//           engine, {buffer.pointer, buffer.length}, &key));
// }

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

      KeyWrapper<::concretelang::clientlib::LweKeyswitchKey> kskw(
          context->getKeys().getKeyswitchKeys());
      KeyWrapper<::concretelang::clientlib::LweBootstrapKey> bskw(
          context->getKeys().getBootstrapKeys());
      KeyWrapper<::concretelang::clientlib::PackingKeyswitchKey> pkskw(
          context->getKeys().getPackingKeyswitchKeys());
      hpx::collectives::broadcast_to("ksk_keystore", kskw);
      hpx::collectives::broadcast_to("bsk_keystore", bskw);
      hpx::collectives::broadcast_to("pksk_keystore", pkskw);
    } else {
      auto kskFut = hpx::collectives::broadcast_from<
          KeyWrapper<::concretelang::clientlib::LweKeyswitchKey>>(
          "ksk_keystore");
      auto bskFut = hpx::collectives::broadcast_from<
          KeyWrapper<::concretelang::clientlib::LweBootstrapKey>>(
          "bsk_keystore");
      auto pkskFut = hpx::collectives::broadcast_from<
          KeyWrapper<::concretelang::clientlib::PackingKeyswitchKey>>(
          "pksk_keystore");

      KeyWrapper<::concretelang::clientlib::LweKeyswitchKey> kskw =
          kskFut.get();
      KeyWrapper<::concretelang::clientlib::LweBootstrapKey> bskw =
          bskFut.get();
      KeyWrapper<::concretelang::clientlib::PackingKeyswitchKey> pkskw =
          pkskFut.get();
      context = new mlir::concretelang::RuntimeContext(
          ::concretelang::clientlib::EvaluationKeys(kskw.keys, bskw.keys,
                                                    pkskw.keys));
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

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
static KeyManager<LweBootstrapKey_u64> *_dfr_node_level_bsk_manager;
static KeyManager<LweKeyswitchKey_u64> *_dfr_node_level_ksk_manager;
static RuntimeContextManager *_dfr_node_level_runtime_context_manager;
} // namespace

void _dfr_register_bsk(LweBootstrapKey_u64 *key, uint64_t key_id);
void _dfr_register_ksk(LweKeyswitchKey_u64 *key, uint64_t key_id);

template <typename LweKeyType> struct KeyWrapper {
  LweKeyType *key;

  KeyWrapper() : key(nullptr) {}
  KeyWrapper(LweKeyType *key) : key(key) {}
  KeyWrapper(KeyWrapper &&moved) noexcept : key(moved.key) {}
  KeyWrapper(const KeyWrapper &kw) : key(kw.key) {}
  friend class hpx::serialization::access;
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const;
  template <class Archive> void load(Archive &ar, const unsigned int version);
  HPX_SERIALIZATION_SPLIT_MEMBER()
};

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

template <typename LweKeyType> struct KeyManager {
  KeyManager() {}
  LweKeyType *get_key(hpx::naming::id_type loc, const uint64_t key_id);

  KeyWrapper<LweKeyType> fetch_key(const uint64_t key_id) {
    std::lock_guard<std::mutex> guard(keystore_guard);

    auto keyit = keystore.find(key_id);
    if (keyit != keystore.end())
      return keyit->second;
    // If this node does not contain this key, this is an error
    // (location was supplied as source for this key).
    HPX_THROW_EXCEPTION(
        hpx::no_success, "fetch_key",
        "Error: could not find key to be fetched on source location.");
  }

  void register_key(LweKeyType *key, uint64_t key_id) {
    std::lock_guard<std::mutex> guard(keystore_guard);
    auto keyit = keystore.find(key_id);
    if (keyit == keystore.end()) {
      keyit = keystore
                  .insert(std::pair<uint64_t, KeyWrapper<LweKeyType>>(
                      key_id, KeyWrapper<LweKeyType>(key)))
                  .first;
      if (keyit == keystore.end()) {
        HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_register_key",
                            "Error: could not register new key.");
      }
    }
  }

  void clear_keys() {
    std::lock_guard<std::mutex> guard(keystore_guard);
    keystore.clear();
  }

private:
  std::mutex keystore_guard;
  std::map<uint64_t, KeyWrapper<LweKeyType>> keystore;
};

KeyWrapper<LweBootstrapKey_u64> _dfr_fetch_bsk(uint64_t key_id) {
  return _dfr_node_level_bsk_manager->fetch_key(key_id);
}

KeyWrapper<LweKeyswitchKey_u64> _dfr_fetch_ksk(uint64_t key_id) {
  return _dfr_node_level_ksk_manager->fetch_key(key_id);
}

} // namespace dfr
} // namespace concretelang
} // namespace mlir

HPX_PLAIN_ACTION(mlir::concretelang::dfr::_dfr_fetch_ksk, _dfr_fetch_ksk_action)
HPX_PLAIN_ACTION(mlir::concretelang::dfr::_dfr_fetch_bsk, _dfr_fetch_bsk_action)

namespace mlir {
namespace concretelang {
namespace dfr {

template <> KeyManager<LweBootstrapKey_u64>::KeyManager() {
  _dfr_node_level_bsk_manager = this;
}

template <>
LweBootstrapKey_u64 *
KeyManager<LweBootstrapKey_u64>::get_key(hpx::naming::id_type loc,
                                         const uint64_t key_id) {
  keystore_guard.lock();
  auto keyit = keystore.find(key_id);
  keystore_guard.unlock();

  if (keyit == keystore.end()) {
    _dfr_fetch_bsk_action fetch;
    KeyWrapper<LweBootstrapKey_u64> &&bskw = fetch(loc, key_id);
    if (bskw.key == nullptr) {
      HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_get_key",
                          "Error: Bootstrap key not found on root node.");
    } else {
      _dfr_register_bsk(bskw.key, key_id);
    }
    return bskw.key;
  }
  return keyit->second.key;
}

template <> KeyManager<LweKeyswitchKey_u64>::KeyManager() {
  _dfr_node_level_ksk_manager = this;
}

template <>
LweKeyswitchKey_u64 *
KeyManager<LweKeyswitchKey_u64>::get_key(hpx::naming::id_type loc,
                                         const uint64_t key_id) {
  keystore_guard.lock();
  auto keyit = keystore.find(key_id);
  keystore_guard.unlock();

  if (keyit == keystore.end()) {
    _dfr_fetch_ksk_action fetch;
    KeyWrapper<LweKeyswitchKey_u64> &&kskw = fetch(loc, key_id);
    if (kskw.key == nullptr) {
      HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_get_key",
                          "Error: Keyswitching key not found on root node.");
    } else {
      _dfr_register_ksk(kskw.key, key_id);
    }
    return kskw.key;
  }
  return keyit->second.key;
}

/************************/
/* Key management API.  */
/************************/

void _dfr_register_bsk(LweBootstrapKey_u64 *key, uint64_t key_id) {
  _dfr_node_level_bsk_manager->register_key(key, key_id);
}
void _dfr_register_ksk(LweKeyswitchKey_u64 *key, uint64_t key_id) {
  _dfr_node_level_ksk_manager->register_key(key, key_id);
}

LweBootstrapKey_u64 *_dfr_get_bsk(hpx::naming::id_type loc, uint64_t key_id) {
  return _dfr_node_level_bsk_manager->get_key(loc, key_id);
}
LweKeyswitchKey_u64 *_dfr_get_ksk(hpx::naming::id_type loc, uint64_t key_id) {
  return _dfr_node_level_ksk_manager->get_key(loc, key_id);
}

/************************/
/* Context management.  */
/************************/

struct RuntimeContextManager {
  // TODO: this is only ok so long as we don't change keys. Once we
  // use multiple keys, should have a map.
  RuntimeContext *context;
  std::mutex context_guard;
  uint64_t ksk_id;
  uint64_t bsk_id;

  RuntimeContextManager() {
    ksk_id = 0;
    bsk_id = 0;
    context = nullptr;
    _dfr_node_level_runtime_context_manager = this;
  }

  RuntimeContext *getContext(uint64_t ksk, uint64_t bsk,
                             hpx::naming::id_type source_locality) {
    std::cout << "GetContext on node " << hpx::get_locality_id()
              << " with context " << context << " " << bsk_id << " " << ksk_id
              << "\n"
              << std::flush;
    if (context != nullptr) {
      std::cout << "simil " << ksk_id << " " << ksk << " " << bsk_id << " "
                << bsk << "\n"
                << std::flush;
      assert(ksk == ksk_id && bsk == bsk_id &&
             "Context manager can only used with single keys for now.");
    } else {
      assert(ksk_id == 0 && bsk_id == 0 &&
             "Context empty but context manager has key ids.");
      LweKeyswitchKey_u64 *keySwitchKey = _dfr_get_ksk(source_locality, ksk);
      LweBootstrapKey_u64 *bootstrapKey = _dfr_get_bsk(source_locality, bsk);
      std::lock_guard<std::mutex> guard(context_guard);
      if (context == nullptr) {
        auto ctx = new RuntimeContext();
        ctx->evaluationKeys = ::concretelang::clientlib::EvaluationKeys(
            std::shared_ptr<::concretelang::clientlib::LweKeyswitchKey>(
                new ::concretelang::clientlib::LweKeyswitchKey(keySwitchKey)),
            std::shared_ptr<::concretelang::clientlib::LweBootstrapKey>(
                new ::concretelang::clientlib::LweBootstrapKey(bootstrapKey)));
        ksk_id = ksk;
        bsk_id = bsk;
        context = ctx;
        std::cout << "Fetching Key ids " << ksk_id << " " << bsk_id << "\n"
                  << std::flush;
      } else {
        std::cout << " GOT context after LOCK on node "
                  << hpx::get_locality_id() << " with context " << context
                  << " " << bsk_id << " " << ksk_id << "\n"
                  << std::flush;
      }
    }
    return context;
  }

  RuntimeContext **getContextAddress() { return &context; }
};

} // namespace dfr
} // namespace concretelang
} // namespace mlir
#endif

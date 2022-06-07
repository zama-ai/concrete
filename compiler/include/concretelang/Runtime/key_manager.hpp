// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DFR_KEY_MANAGER_HPP
#define CONCRETELANG_DFR_KEY_MANAGER_HPP

#include <memory>
#include <utility>

#include <hpx/include/runtime.hpp>
#include <hpx/modules/collectives.hpp>

#include "concretelang/Runtime/DFRuntime.hpp"

struct PbsKeyManager;
extern PbsKeyManager *node_level_key_manager;

struct PbsKeyWrapper {
  std::shared_ptr<void *> key;
  size_t key_id;
  size_t size;

  PbsKeyWrapper() {}

  PbsKeyWrapper(void *key, size_t key_id, size_t size)
      : key(std::make_shared<void *>(key)), key_id(key_id), size(size) {}

  PbsKeyWrapper(std::shared_ptr<void *> key, size_t key_id, size_t size)
      : key(key), key_id(key_id), size(size) {}

  PbsKeyWrapper(PbsKeyWrapper &&moved) noexcept
      : key(moved.key), key_id(moved.key_id), size(moved.size) {}

  PbsKeyWrapper(const PbsKeyWrapper &pbsk)
      : key(pbsk.key), key_id(pbsk.key_id), size(pbsk.size) {}

  friend class hpx::serialization::access;
  template <class Archive>
  void save(Archive &ar, const unsigned int version) const {
    char *_key_ = static_cast<char *>(*key);
    ar &key_id &size;
    for (size_t i = 0; i < size; ++i)
      ar &_key_[i];
  }

  template <class Archive> void load(Archive &ar, const unsigned int version) {
    ar &key_id &size;
    char *_key_ = (char *)malloc(size);
    for (size_t i = 0; i < size; ++i)
      ar &_key_[i];
    key = std::make_shared<void *>(_key_);
  }
  HPX_SERIALIZATION_SPLIT_MEMBER()
};

inline bool operator==(const PbsKeyWrapper &lhs, const PbsKeyWrapper &rhs) {
  return lhs.key_id == rhs.key_id;
}

PbsKeyWrapper _dfr_fetch_key(size_t);
HPX_PLAIN_ACTION(_dfr_fetch_key, _dfr_fetch_key_action)

struct PbsKeyManager {
  // The initial keys registered on the root node and whether to push
  // them is TBD.

  PbsKeyManager() { node_level_key_manager = this; }

  PbsKeyWrapper get_key(const size_t key_id) {
    keystore_guard.lock();
    auto keyit = keystore.find(key_id);
    keystore_guard.unlock();

    if (keyit == keystore.end()) {
      _dfr_fetch_key_action fet;
      PbsKeyWrapper &&pkw = fet(hpx::find_root_locality(), key_id);
      if (pkw.size == 0) {
        // Maybe retry or try other nodes... but for now it's an error.
        HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_get_key",
                            "Error: key not found on remote node.");
      } else {
        std::lock_guard<std::mutex> guard(keystore_guard);
        keyit = keystore.insert(std::pair<size_t, PbsKeyWrapper>(key_id, pkw))
                    .first;
      }
    }
    return keyit->second;
  }

  // To be used only for remote requests
  PbsKeyWrapper fetch_key(const size_t key_id) {
    std::lock_guard<std::mutex> guard(keystore_guard);

    auto keyit = keystore.find(key_id);
    if (keyit != keystore.end())
      return keyit->second;
    // If this node does not contain this key, return an empty wrapper
    return PbsKeyWrapper(nullptr, 0, 0);
  }

  void register_key(void *key, size_t key_id, size_t size) {
    std::lock_guard<std::mutex> guard(keystore_guard);
    auto keyit = keystore
                     .insert(std::pair<size_t, PbsKeyWrapper>(
                         key_id, PbsKeyWrapper(key, key_id, size)))
                     .first;
    if (keyit == keystore.end()) {
      HPX_THROW_EXCEPTION(hpx::no_success, "_dfr_register_key",
                          "Error: could not register new key.");
    }
  }

  void broadcast_keys() {
    std::lock_guard<std::mutex> guard(keystore_guard);
    if (_dfr_is_root_node())
      hpx::collectives::broadcast_to("keystore", this->keystore).get();
    else
      keystore = std::move(
          hpx::collectives::broadcast_from<std::map<size_t, PbsKeyWrapper>>(
              "keystore")
              .get());
  }

private:
  std::mutex keystore_guard;
  std::map<size_t, PbsKeyWrapper> keystore;
};

PbsKeyWrapper _dfr_fetch_key(size_t key_id) {
  return node_level_key_manager->fetch_key(key_id);
}

#endif

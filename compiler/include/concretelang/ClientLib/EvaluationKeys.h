// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_EVALUATION_KEYS_H_
#define CONCRETELANG_CLIENTLIB_EVALUATION_KEYS_H_

#include <memory>

extern "C" {
#include "concrete-ffi.h"
}

namespace concretelang {
namespace clientlib {

// =============================================

/// Wrapper for `LweKeyswitchKey_u64` so that it cleans up properly.
class LweKeyswitchKey {
private:
  LweKeyswitchKey_u64 *ksk;

protected:
  friend std::ostream &operator<<(std::ostream &ostream,
                                  const LweKeyswitchKey &wrappedKsk);
  friend std::istream &operator>>(std::istream &istream,
                                  LweKeyswitchKey &wrappedKsk);

public:
  LweKeyswitchKey(LweKeyswitchKey_u64 *ksk) : ksk{ksk} {}
  LweKeyswitchKey(LweKeyswitchKey &other) = delete;
  LweKeyswitchKey(LweKeyswitchKey &&other) : ksk{other.ksk} {
    other.ksk = nullptr;
  }
  ~LweKeyswitchKey() {
    if (this->ksk != nullptr) {
      free_lwe_keyswitch_key_u64(this->ksk);
      this->ksk = nullptr;
    }
  }

  LweKeyswitchKey_u64 *get() { return this->ksk; }
};

// =============================================

/// Wrapper for `LweBootstrapKey_u64` so that it cleans up properly.
class LweBootstrapKey {
private:
  LweBootstrapKey_u64 *bsk;

protected:
  friend std::ostream &operator<<(std::ostream &ostream,
                                  const LweBootstrapKey &wrappedBsk);
  friend std::istream &operator>>(std::istream &istream,
                                  LweBootstrapKey &wrappedBsk);

public:
  LweBootstrapKey(LweBootstrapKey_u64 *bsk) : bsk{bsk} {}
  LweBootstrapKey(LweBootstrapKey &other) = delete;
  LweBootstrapKey(LweBootstrapKey &&other) : bsk{other.bsk} {
    other.bsk = nullptr;
  }
  ~LweBootstrapKey() {
    if (this->bsk != nullptr) {
      free_lwe_bootstrap_key_u64(this->bsk);
      this->bsk = nullptr;
    }
  }

  LweBootstrapKey_u64 *get() { return this->bsk; }
};

// =============================================

/// Evalution keys required for execution.
class EvaluationKeys {
private:
  std::shared_ptr<LweKeyswitchKey> sharedKsk;
  std::shared_ptr<LweBootstrapKey> sharedBsk;

protected:
  friend std::ostream &operator<<(std::ostream &ostream,
                                  const EvaluationKeys &evaluationKeys);
  friend std::istream &operator>>(std::istream &istream,
                                  EvaluationKeys &evaluationKeys);

public:
  EvaluationKeys()
      : sharedKsk{std::shared_ptr<LweKeyswitchKey>(nullptr)},
        sharedBsk{std::shared_ptr<LweBootstrapKey>(nullptr)} {}

  EvaluationKeys(std::shared_ptr<LweKeyswitchKey> sharedKsk,
                 std::shared_ptr<LweBootstrapKey> sharedBsk)
      : sharedKsk{sharedKsk}, sharedBsk{sharedBsk} {}

  LweKeyswitchKey_u64 *getKsk() { return this->sharedKsk->get(); }
  LweBootstrapKey_u64 *getBsk() { return this->sharedBsk->get(); }
};

// =============================================

} // namespace clientlib
} // namespace concretelang

#endif

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_EVALUATION_KEYS_H_
#define CONCRETELANG_CLIENTLIB_EVALUATION_KEYS_H_

#include <memory>

#include "concrete-core-ffi.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

// =============================================

/// Wrapper for `LweKeyswitchKey64` so that it cleans up properly.
class LweKeyswitchKey {
private:
  LweKeyswitchKey64 *ksk;

protected:
  friend std::ostream &operator<<(std::ostream &ostream,
                                  const LweKeyswitchKey &wrappedKsk);
  friend std::istream &operator>>(std::istream &istream,
                                  LweKeyswitchKey &wrappedKsk);

public:
  LweKeyswitchKey(LweKeyswitchKey64 *ksk) : ksk{ksk} {}
  LweKeyswitchKey(LweKeyswitchKey &other) = delete;
  LweKeyswitchKey(LweKeyswitchKey &&other) : ksk{other.ksk} {
    other.ksk = nullptr;
  }
  ~LweKeyswitchKey() {
    if (this->ksk != nullptr) {
      CAPI_ASSERT_ERROR(destroy_lwe_keyswitch_key_u64(this->ksk));

      this->ksk = nullptr;
    }
  }

  LweKeyswitchKey64 *get() { return this->ksk; }
};

// =============================================

/// Wrapper for `FftwFourierLweBootstrapKey64` so that it cleans up properly.
class LweBootstrapKey {
private:
  FftwFourierLweBootstrapKey64 *bsk;

protected:
  friend std::ostream &operator<<(std::ostream &ostream,
                                  const LweBootstrapKey &wrappedBsk);
  friend std::istream &operator>>(std::istream &istream,
                                  LweBootstrapKey &wrappedBsk);

public:
  LweBootstrapKey(FftwFourierLweBootstrapKey64 *bsk) : bsk{bsk} {}
  LweBootstrapKey(LweBootstrapKey &other) = delete;
  LweBootstrapKey(LweBootstrapKey &&other) : bsk{other.bsk} {
    other.bsk = nullptr;
  }
  ~LweBootstrapKey() {
    if (this->bsk != nullptr) {
      CAPI_ASSERT_ERROR(destroy_fftw_fourier_lwe_bootstrap_key_u64(this->bsk));
      this->bsk = nullptr;
    }
  }

  FftwFourierLweBootstrapKey64 *get() { return this->bsk; }
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

  LweKeyswitchKey64 *getKsk() { return this->sharedKsk->get(); }
  FftwFourierLweBootstrapKey64 *getBsk() { return this->sharedBsk->get(); }
};

// =============================================

} // namespace clientlib
} // namespace concretelang

#endif

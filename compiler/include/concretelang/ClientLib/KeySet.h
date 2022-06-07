// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_KEYSET_H_
#define CONCRETELANG_CLIENTLIB_KEYSET_H_

#include <memory>

#include "boost/outcome.h"

extern "C" {
#include "concrete-ffi.h"
}
#include "concretelang/Runtime/context.h"

#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/Common/Error.h"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;
using RuntimeContext = mlir::concretelang::RuntimeContext;

class KeySet {
public:
  KeySet();
  ~KeySet();
  KeySet(KeySet &other) = delete;

  // allocate a KeySet according the ClientParameters.
  static outcome::checked<std::unique_ptr<KeySet>, StringError>
  generate(ClientParameters &params, uint64_t seed_msb, uint64_t seed_lsb);

  // isInputEncrypted return true if the input at the given pos is encrypted.
  bool isInputEncrypted(size_t pos);

  // getInputLweSecretKeyParam returns the parameters of the lwe secret key for
  // the input at the given `pos`.
  // The input must be encrupted
  LweSecretKeyParam getInputLweSecretKeyParam(size_t pos) {
    auto gate = inputGate(pos);
    auto inputSk = this->secretKeys.find(gate.encryption->secretKeyID);
    return inputSk->second.first;
  }

  // getOutputLweSecretKeyParam returns the parameters of the lwe secret key for
  // the given output.
  LweSecretKeyParam getOutputLweSecretKeyParam(size_t pos) {
    auto gate = outputGate(pos);
    auto outputSk = this->secretKeys.find(gate.encryption->secretKeyID);
    return outputSk->second.first;
  }

  // allocate a lwe ciphertext buffer for the argument at argPos, set the size
  // of the allocated buffer.
  outcome::checked<void, StringError>
  allocate_lwe(size_t argPos, uint64_t **ciphertext, uint64_t &size);

  // encrypt the input to the ciphertext for the argument at argPos.
  outcome::checked<void, StringError>
  encrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t input);

  // isOuputEncrypted return true if the output at the given pos is encrypted.
  bool isOutputEncrypted(size_t pos);

  // decrypt the ciphertext to the output for the argument at argPos.
  outcome::checked<void, StringError>
  decrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t &output);

  size_t numInputs() { return inputs.size(); }
  size_t numOutputs() { return outputs.size(); }

  CircuitGate inputGate(size_t pos) { return std::get<0>(inputs[pos]); }
  CircuitGate outputGate(size_t pos) { return std::get<0>(outputs[pos]); }

  RuntimeContext runtimeContext() {
    RuntimeContext context;
    context.evaluationKeys = this->evaluationKeys();
    return context;
  }

  EvaluationKeys evaluationKeys() {
    auto sharedKsk = std::get<1>(this->keyswitchKeys.at("ksk_v0"));
    auto sharedBsk = std::get<1>(this->bootstrapKeys.at("bsk_v0"));
    return EvaluationKeys(sharedKsk, sharedBsk);
  }

  const std::map<LweSecretKeyID,
                 std::pair<LweSecretKeyParam, LweSecretKey_u64 *>> &
  getSecretKeys();

  const std::map<LweSecretKeyID,
                 std::pair<BootstrapKeyParam, std::shared_ptr<LweBootstrapKey>>>
      &getBootstrapKeys();

  const std::map<LweSecretKeyID,
                 std::pair<KeyswitchKeyParam, std::shared_ptr<LweKeyswitchKey>>>
      &getKeyswitchKeys();

protected:
  outcome::checked<void, StringError>
  generateSecretKey(LweSecretKeyID id, LweSecretKeyParam param);

  outcome::checked<void, StringError>
  generateBootstrapKey(BootstrapKeyID id, BootstrapKeyParam param);

  outcome::checked<void, StringError>
  generateKeyswitchKey(KeyswitchKeyID id, KeyswitchKeyParam param);

  outcome::checked<void, StringError>
  generateKeysFromParams(ClientParameters &params, uint64_t seed_msb,
                         uint64_t seed_lsb);

  outcome::checked<void, StringError>
  setupEncryptionMaterial(ClientParameters &params, uint64_t seed_msb,
                          uint64_t seed_lsb);

  friend class KeySetCache;

private:
  Engine *engine;
  std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
      secretKeys;
  std::map<LweSecretKeyID,
           std::pair<BootstrapKeyParam, std::shared_ptr<LweBootstrapKey>>>
      bootstrapKeys;
  std::map<LweSecretKeyID,
           std::pair<KeyswitchKeyParam, std::shared_ptr<LweKeyswitchKey>>>
      keyswitchKeys;
  std::vector<std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey_u64 *>>
      inputs;
  std::vector<std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey_u64 *>>
      outputs;

  void setKeys(
      std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
          secretKeys,
      std::map<LweSecretKeyID,
               std::pair<BootstrapKeyParam, std::shared_ptr<LweBootstrapKey>>>
          bootstrapKeys,
      std::map<LweSecretKeyID,
               std::pair<KeyswitchKeyParam, std::shared_ptr<LweKeyswitchKey>>>
          keyswitchKeys);
};

} // namespace clientlib
} // namespace concretelang

#endif

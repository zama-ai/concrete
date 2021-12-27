// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_SUPPORT_KEYSET_H_
#define ZAMALANG_SUPPORT_KEYSET_H_

#include "llvm/Support/Error.h"
#include <memory>

extern "C" {
#include "concrete-ffi.h"
#include "zamalang/Runtime/context.h"
}

#include "zamalang/Support/ClientParameters.h"
#include "zamalang/Support/KeySetCache.h"

namespace mlir {
namespace zamalang {

class KeySet {
public:
  ~KeySet();

  static std::unique_ptr<KeySet> uninitialized();

  llvm::Error generateKeysFromParams(ClientParameters &params,
                                     uint64_t seed_msb, uint64_t seed_lsb);

  llvm::Error setupEncryptionMaterial(ClientParameters &params,
                                      uint64_t seed_msb, uint64_t seed_lsb);

  // allocate a KeySet according the ClientParameters.
  static llvm::Expected<std::unique_ptr<KeySet>>
  generate(ClientParameters &params, uint64_t seed_msb, uint64_t seed_lsb);

  // isInputEncrypted return true if the input at the given pos is encrypted.
  bool isInputEncrypted(size_t pos);
  // allocate a lwe ciphertext for the argument at argPos.
  llvm::Error allocate_lwe(size_t argPos, LweCiphertext_u64 **ciphertext);
  // encrypt the input to the ciphertext for the argument at argPos.
  llvm::Error encrypt_lwe(size_t argPos, LweCiphertext_u64 *ciphertext,
                          uint64_t input);

  // isOuputEncrypted return true if the output at the given pos is encrypted.
  bool isOutputEncrypted(size_t pos);
  // decrypt the ciphertext to the output for the argument at argPos.
  llvm::Error decrypt_lwe(size_t argPos, LweCiphertext_u64 *ciphertext,
                          uint64_t &output);

  size_t numInputs() { return inputs.size(); }
  size_t numOutputs() { return outputs.size(); }

  CircuitGate inputGate(size_t pos) { return std::get<0>(inputs[pos]); }
  CircuitGate outputGate(size_t pos) { return std::get<0>(outputs[pos]); }

  void setRuntimeContext(RuntimeContext &context) {
    context.ksk = std::get<1>(this->keyswitchKeys["ksk_v0"]);
    context.bsk = std::get<1>(this->bootstrapKeys["bsk_v0"]);
  }

  const std::map<LweSecretKeyID,
                 std::pair<LweSecretKeyParam, LweSecretKey_u64 *>> &
  getSecretKeys();

  const std::map<LweSecretKeyID,
                 std::pair<BootstrapKeyParam, LweBootstrapKey_u64 *>> &
  getBootstrapKeys();

  const std::map<LweSecretKeyID,
                 std::pair<KeyswitchKeyParam, LweKeyswitchKey_u64 *>> &
  getKeyswitchKeys();

protected:
  llvm::Error generateSecretKey(LweSecretKeyID id, LweSecretKeyParam param,
                                SecretRandomGenerator *generator);
  llvm::Error generateBootstrapKey(BootstrapKeyID id, BootstrapKeyParam param,
                                   EncryptionRandomGenerator *generator);
  llvm::Error generateKeyswitchKey(KeyswitchKeyID id, KeyswitchKeyParam param,
                                   EncryptionRandomGenerator *generator);

  friend class KeySetCache;

private:
  EncryptionRandomGenerator *encryptionRandomGenerator;
  std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
      secretKeys;
  std::map<LweSecretKeyID, std::pair<BootstrapKeyParam, LweBootstrapKey_u64 *>>
      bootstrapKeys;
  std::map<LweSecretKeyID, std::pair<KeyswitchKeyParam, LweKeyswitchKey_u64 *>>
      keyswitchKeys;
  std::vector<std::tuple<CircuitGate, LweSecretKeyParam *, LweSecretKey_u64 *>>
      inputs;
  std::vector<std::tuple<CircuitGate, LweSecretKeyParam *, LweSecretKey_u64 *>>
      outputs;

  void setKeys(
      std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
          secretKeys,
      std::map<LweSecretKeyID,
               std::pair<BootstrapKeyParam, LweBootstrapKey_u64 *>>
          bootstrapKeys,
      std::map<LweSecretKeyID,
               std::pair<KeyswitchKeyParam, LweKeyswitchKey_u64 *>>
          keyswitchKeys);
};

} // namespace zamalang
} // namespace mlir

#endif
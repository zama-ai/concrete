// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_KEYSET_H_
#define CONCRETELANG_CLIENTLIB_KEYSET_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "boost/outcome.h"
#ifdef OUTPUT_COMPRESSION_SUPPORT
#include "CompressLWE/defines.h"
#endif
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/KeySetCache.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/DFRuntime.hpp"

namespace concretelang {
namespace clientlib {

using concretelang::error::StringError;

class KeySet {
public:
  KeySet(ClientParameters clientParameters, CSPRNG &&csprng)
      : csprng(std::move(csprng)), _clientParameters(clientParameters){};
  KeySet(KeySet &other) = delete;

  /// Generate a KeySet from a ClientParameters specification.
  static outcome::checked<std::unique_ptr<KeySet>, StringError>
  generate(ClientParameters clientParameters, CSPRNG &&csprng);

  /// Create a KeySet from a set of given keys
  static outcome::checked<std::unique_ptr<KeySet>, StringError>
  fromKeys(ClientParameters clientParameters,
           std::vector<LweSecretKey> secretKeys,
           std::vector<LweBootstrapKey> bootstrapKeys,
           std::vector<LweKeyswitchKey> keyswitchKeys,
           std::vector<PackingKeyswitchKey> packingKeyswitchKeys,
#ifdef OUTPUT_COMPRESSION_SUPPORT
           std::optional<PaiFullKeys> paiFullKeys,
#endif
           CSPRNG &&csprng);

  /// Returns the ClientParameters associated with the KeySet.
  ClientParameters clientParameters() const { return _clientParameters; }

  // isInputEncrypted return true if the input at the given pos is encrypted.
  bool isInputEncrypted(size_t pos);

  /// allocate a lwe ciphertext buffer for the argument at argPos, set the size
  /// of the allocated buffer.
  outcome::checked<void, StringError>
  allocate_lwe(size_t argPos, uint64_t **ciphertext, uint64_t &size);

  /// encrypt the input to the ciphertext for the argument at argPos.
  outcome::checked<void, StringError>
  encode_encrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t input);

  /// isOuputEncrypted return true if the output at the given pos is encrypted.
  bool isOutputEncrypted(size_t pos);

  /// decrypt the ciphertext to the output for the argument at argPos.
  outcome::checked<uint64_t, StringError>
  decrypt_lwe(size_t argPos, uint64_t *ciphertext) const;

  outcome::checked<void, StringError>
  decrypt_decode_lwe(size_t argPos, uint64_t *ciphertext,
                     uint64_t &output) const;

  size_t numInputs() const { return inputs.size(); }
  size_t numOutputs() const { return outputs.size(); }

  CircuitGate inputGate(size_t pos) const { return std::get<0>(inputs[pos]); }
  CircuitGate outputGate(size_t pos) const { return std::get<0>(outputs[pos]); }

  /// @brief evaluationKeys returns the evaluation keys associate to this client
  /// keyset. Those evaluations keys can be safely shared publicly
  EvaluationKeys evaluationKeys();

  const std::vector<LweSecretKey> &getSecretKeys() const;

  const std::vector<LweBootstrapKey> &getBootstrapKeys() const;

  const std::vector<LweKeyswitchKey> &getKeyswitchKeys() const;

  const std::vector<PackingKeyswitchKey> &getPackingKeyswitchKeys() const;

#ifdef OUTPUT_COMPRESSION_SUPPORT
  const std::optional<PaiFullKeys> &getPaiFullKey() const;
#endif

protected:
  outcome::checked<void, StringError>
  generateSecretKey(LweSecretKeyParam param);

  outcome::checked<void, StringError>
  generateBootstrapKey(BootstrapKeyParam param);

  outcome::checked<void, StringError>
  generateKeyswitchKey(KeyswitchKeyParam param);

  outcome::checked<void, StringError>
  generatePackingKeyswitchKey(PackingKeyswitchKeyParam param);

#ifdef OUTPUT_COMPRESSION_SUPPORT
  outcome::checked<void, StringError> generatePaiKeys(PaiKeyParam param);
#endif

  outcome::checked<void, StringError> generateKeysFromParams();

  outcome::checked<void, StringError> setupEncryptionMaterial();

  friend class KeySetCache;

private:
  CSPRNG csprng;

  ///////////////////////////////////////////////
  // Keys mappings
  std::vector<LweSecretKey> secretKeys;
  std::vector<LweBootstrapKey> bootstrapKeys;
  std::vector<LweKeyswitchKey> keyswitchKeys;
  std::vector<PackingKeyswitchKey> packingKeyswitchKeys;
#ifdef OUTPUT_COMPRESSION_SUPPORT
  std::optional<PaiFullKeys> paiFullKeys;
#endif

  outcome::checked<LweSecretKey, StringError> findLweSecretKey(LweSecretKeyID);

  ///////////////////////////////////////////////
  // Convenient positional mapping between positional gate en secret key
  typedef std::vector<std::pair<CircuitGate, std::optional<LweSecretKey>>>
      SecretKeyInputGateMapping;
  outcome::checked<SecretKeyInputGateMapping, StringError>
  mapCircuitInputGateLweSecretKey(std::vector<CircuitGate> gates);

  typedef std::vector<std::tuple<CircuitGate, std::optional<LweSecretKey>
#ifdef OUTPUT_COMPRESSION_SUPPORT
                                 ,
                                 std::optional<PaiFullKeys>
#endif
                                 >>
      SecretKeyOutputGateMapping;
  outcome::checked<SecretKeyOutputGateMapping, StringError>
  mapCircuitOutputGateLweSecretKey(std::vector<CircuitGate> gates);

  SecretKeyInputGateMapping inputs;
  SecretKeyOutputGateMapping outputs;

  clientlib::ClientParameters _clientParameters;
};

} // namespace clientlib
} // namespace concretelang

#endif

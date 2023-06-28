// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/KeySet.h"

#ifdef OUTPUT_COMPRESSION_SUPPORT
#include "compress_lwe/defines.h"
#include "compress_lwe/library.h"
#endif
#include "concretelang/ClientLib/CRT.h"
#include "concretelang/ClientLib/ClientParameters.h"
#include "concretelang/ClientLib/EvaluationKeys.h"
#include "concretelang/ClientLib/PublicArguments.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Support/Error.h"
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <optional>

namespace concretelang {
namespace clientlib {

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySet::generate(ClientParameters clientParameters, CSPRNG &&csprng) {
  auto keySet = std::make_unique<KeySet>(clientParameters, std::move(csprng));
  OUTCOME_TRYV(keySet->generateKeysFromParams());
  OUTCOME_TRYV(keySet->setupEncryptionMaterial());
  return std::move(keySet);
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySet::fromKeys(ClientParameters clientParameters,
                 std::vector<LweSecretKey> secretKeys,
                 std::vector<LweBootstrapKey> bootstrapKeys,
                 std::vector<LweKeyswitchKey> keyswitchKeys,
                 std::vector<PackingKeyswitchKey> packingKeyswitchKeys,
#ifdef OUTPUT_COMPRESSION_SUPPORT
                 std::optional<comp::FullKeys> fullKeys,
#endif
                 CSPRNG &&csprng) {

  auto keySet = std::make_unique<KeySet>(clientParameters, std::move(csprng));
  keySet->secretKeys = secretKeys;
  keySet->bootstrapKeys = bootstrapKeys;
  keySet->keyswitchKeys = keyswitchKeys;
  keySet->packingKeyswitchKeys = packingKeyswitchKeys;
#ifdef OUTPUT_COMPRESSION_SUPPORT
  keySet->fullKeys = std::move(fullKeys);
#endif
  OUTCOME_TRYV(keySet->setupEncryptionMaterial());
  return std::move(keySet);
}

EvaluationKeys KeySet::evaluationKeys() {
#ifdef OUTPUT_COMPRESSION_SUPPORT

  std::optional<comp::CompressionKey> compressionKeys;

  if (fullKeys.has_value()) {
    compressionKeys = fullKeys->compression_key();
  }
#endif
  return EvaluationKeys(keyswitchKeys, bootstrapKeys, packingKeyswitchKeys
#ifdef OUTPUT_COMPRESSION_SUPPORT
                        ,
                        compressionKeys
#endif
  );
}

outcome::checked<KeySet::SecretKeyInputGateMapping, StringError>
KeySet::mapCircuitInputGateLweSecretKey(std::vector<CircuitGate> gates) {
  SecretKeyInputGateMapping mapping;
  for (auto gate : gates) {
    std::pair<CircuitGate, std::optional<LweSecretKey>> input;
    if (gate.encryption.has_value()) {
      assert(gate.encryption->secretKeyID < this->secretKeys.size());
      auto skIt = this->secretKeys[gate.encryption->secretKeyID];

      input = {gate, skIt};
    } else {
      input = {gate, std::nullopt};
    }
    mapping.push_back(input);
  }
  return mapping;
}

outcome::checked<KeySet::SecretKeyOutputGateMapping, StringError>
KeySet::mapCircuitOutputGateLweSecretKey(std::vector<CircuitGate> gates) {
  SecretKeyOutputGateMapping mapping;
  for (auto gate : gates) {

    std::tuple<CircuitGate, std::optional<LweSecretKey>
#ifdef OUTPUT_COMPRESSION_SUPPORT
               ,
               std::optional<comp::FullKeys>
#endif
               >
        output;
    if (gate.encryption.has_value()) {
      assert(gate.encryption->secretKeyID < this->secretKeys.size());
      auto skIt = this->secretKeys[gate.encryption->secretKeyID];

#ifdef OUTPUT_COMPRESSION_SUPPORT
      comp::FullKeys fullkey;

      if (gate.compression) {
        output = {gate, skIt, std::move(fullkey)};
      } else {
        output = {gate, skIt, std::nullopt};
      }
#else
      output = {gate, skIt};
#endif
    } else {
#ifdef OUTPUT_COMPRESSION_SUPPORT
      output = {gate, std::nullopt, std::nullopt};
#else
      output = {gate, std::nullopt};
#endif
    }
    mapping.push_back(output);
  }
  return mapping;
}

outcome::checked<void, StringError> KeySet::setupEncryptionMaterial() {
  OUTCOME_TRY(this->inputs,
              mapCircuitInputGateLweSecretKey(_clientParameters.inputs));
  OUTCOME_TRY(this->outputs,
              mapCircuitOutputGateLweSecretKey(_clientParameters.outputs));

  return outcome::success();
}

outcome::checked<void, StringError> KeySet::generateKeysFromParams() {

  // Generate LWE secret keys
  for (auto secretKeyParam : _clientParameters.secretKeys) {
    OUTCOME_TRYV(this->generateSecretKey(secretKeyParam));
  }
  // Generate bootstrap keys
  for (auto bootstrapKeyParam : _clientParameters.bootstrapKeys) {
    OUTCOME_TRYV(this->generateBootstrapKey(bootstrapKeyParam));
  }
  // Generate keyswitch key
  for (auto keyswitchParam : _clientParameters.keyswitchKeys) {
    OUTCOME_TRYV(this->generateKeyswitchKey(keyswitchParam));
  }
  // Generate packing keyswitch key
  for (auto packingKeyswitchKeyParam : _clientParameters.packingKeyswitchKeys) {
    OUTCOME_TRYV(this->generatePackingKeyswitchKey(packingKeyswitchKeyParam));
  }
#ifdef OUTPUT_COMPRESSION_SUPPORT

  // Generate compression key
  if (_clientParameters.paiCompKeys.has_value()) {
    OUTCOME_TRYV(this->generatePaiKeys(*_clientParameters.paiCompKeys));
  }
#endif
  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generateSecretKey(LweSecretKeyParam param) {
  // Init the lwe secret key
  LweSecretKey sk(param, csprng);
  // Store the lwe secret key
  secretKeys.push_back(sk);
  return outcome::success();
}

outcome::checked<LweSecretKey, StringError>
KeySet::findLweSecretKey(LweSecretKeyID keyID) {
  assert(keyID < secretKeys.size());
  auto secretKey = secretKeys[keyID];

  return secretKey;
}

outcome::checked<void, StringError>
KeySet::generateBootstrapKey(BootstrapKeyParam param) {
  // Finding input and output secretKeys
  OUTCOME_TRY(auto inputKey, findLweSecretKey(param.inputSecretKeyID));
  OUTCOME_TRY(auto outputKey, findLweSecretKey(param.outputSecretKeyID));
  // Initialize the bootstrap key
  LweBootstrapKey bootstrapKey(param, inputKey, outputKey, csprng);
  // Store the bootstrap key
  bootstrapKeys.push_back(std::move(bootstrapKey));
  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generateKeyswitchKey(KeyswitchKeyParam param) {
  // Finding input and output secretKeys
  OUTCOME_TRY(auto inputKey, findLweSecretKey(param.inputSecretKeyID));
  OUTCOME_TRY(auto outputKey, findLweSecretKey(param.outputSecretKeyID));
  // Initialize the bootstrap key
  LweKeyswitchKey keyswitchKey(param, inputKey, outputKey, csprng);
  // Store the keyswitch key
  keyswitchKeys.push_back(keyswitchKey);
  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generatePackingKeyswitchKey(PackingKeyswitchKeyParam param) {
  // Finding input secretKeys
  assert(param.inputSecretKeyID < secretKeys.size());
  auto inputSk = secretKeys[param.inputSecretKeyID];

  assert(param.outputSecretKeyID < secretKeys.size());
  auto outputSk = secretKeys[param.outputSecretKeyID];

  PackingKeyswitchKey packingKeyswitchKey(param, inputSk, outputSk, csprng);
  // Store the keyswitch key
  packingKeyswitchKeys.push_back(packingKeyswitchKey);
  return outcome::success();
}

#ifdef OUTPUT_COMPRESSION_SUPPORT
outcome::checked<void, StringError> KeySet::generatePaiKeys(PaiKeyParam param) {
  assert(param.secretKeyID < secretKeys.size());
  auto sk = secretKeys[param.secretKeyID];

  fullKeys = comp::generateKeys(sk.vector(), 1);

  return outcome::success();
}
#endif

outcome::checked<void, StringError>
KeySet::allocate_lwe(size_t argPos, uint64_t **ciphertext, uint64_t &size) {
  if (argPos >= inputs.size()) {
    return StringError("allocate_lwe position of argument is too high");
  }
  auto inputSk = inputs[argPos];
  auto encryption = std::get<0>(inputSk).encryption;
  if (!encryption.has_value()) {
    return StringError("allocate_lwe argument #")
           << argPos << "is not encypeted";
  }
  auto numBlocks =
      encryption->encoding.crt.empty() ? 1 : encryption->encoding.crt.size();
  assert(inputSk.second.has_value());

  size = inputSk.second->parameters().lweSize();
  *ciphertext = (uint64_t *)malloc(sizeof(uint64_t) * size * numBlocks);
  return outcome::success();
}

bool KeySet::isInputEncrypted(size_t argPos) {
  return argPos < inputs.size() &&
         std::get<0>(inputs[argPos]).encryption.has_value();
}

bool KeySet::isOutputEncrypted(size_t argPos) {
  return argPos < outputs.size() &&
         std::get<0>(outputs[argPos]).encryption.has_value();
}

/// Return the number of bits to represents the given value
uint64_t bitWidthOfValue(uint64_t value) { return std::ceil(std::log2(value)); }

outcome::checked<void, StringError>
KeySet::encode_encrypt_lwe(size_t argPos, uint64_t *ciphertext,
                           uint64_t input) {
  if (argPos >= inputs.size()) {
    return StringError("encrypt_lwe position of argument is too high");
  }
  const auto &inputSk = inputs[argPos];
  auto encryption = std::get<0>(inputSk).encryption;
  if (!encryption.has_value()) {
    return StringError("encrypt_lwe the positional argument is not encrypted");
  }
  auto encoding = encryption->encoding;
  assert(inputSk.second.has_value());
  auto lweSecretKey = *inputSk.second;
  auto lweSecretKeyParam = lweSecretKey.parameters();
  // CRT encoding - N blocks with crt encoding
  auto crt = encryption->encoding.crt;
  if (!crt.empty()) {
    // Put each decomposition into a new ciphertext
    auto product = crt::productOfModuli(crt);
    for (auto modulus : crt) {
      auto plaintext = crt::encode(input, modulus, product);
      lweSecretKey.encrypt(ciphertext, plaintext, encryption->variance, csprng);
      ciphertext = ciphertext + lweSecretKeyParam.lweSize();
    }
    return outcome::success();
  }
  // Simple TFHE integers - 1 blocks with one padding bits
  // TODO we could check if the input value is in the right range
  uint64_t plaintext = input << (64 - (encryption->encoding.precision + 1));
  lweSecretKey.encrypt(ciphertext, plaintext, encryption->variance, csprng);
  return outcome::success();
}

outcome::checked<uint64_t, StringError>
KeySet::decrypt_lwe(size_t argPos, uint64_t *ciphertext) const {
  if (argPos >= outputs.size()) {
    return StringError("decrypt_lwe: position of argument is too high");
  }
  const auto &outputSk = outputs[argPos];
  assert(std::get<1>(outputSk).has_value());
  auto &lweSecretKey = *std::get<1>(outputSk);
  auto encryption = std::get<0>(outputSk).encryption;
  if (!encryption.has_value()) {
    return StringError("decrypt_lwe: the positional argument is not encrypted");
  }
  uint64_t decrypted = 0;
  lweSecretKey.decrypt(ciphertext, decrypted);

  return decrypted;
}

outcome::checked<void, StringError>
KeySet::decrypt_decode_lwe(size_t argPos, uint64_t *ciphertext,
                           uint64_t &output) const {

  if (argPos >= outputs.size()) {
    return StringError("decrypt_lwe: position of argument is too high");
  }
  const auto &outputSk = outputs[argPos];
  assert(std::get<1>(outputSk).has_value());
  auto &lweSecretKey = *std::get<1>(outputSk);

  auto lweSecretKeyParam = lweSecretKey.parameters();
  auto encryption = std::get<0>(outputSk).encryption;

  auto crt = encryption->encoding.crt;

  if (crt.empty()) {

    uint64_t decrypted = 0;
    lweSecretKey.decrypt(ciphertext, decrypted);

    output = decode_1padded_integer(decrypted, encryption->encoding.precision,
                                    encryption->encoding.isSigned);

  } else {
    // CRT encoded TFHE integers

    // Decrypt and decode remainders
    std::vector<int64_t> remainders;
    uint i = 0;
    for (auto modulus : crt) {
      uint64_t decrypted = 0;
      lweSecretKey.decrypt(ciphertext + i * lweSecretKeyParam.lweSize(),
                           decrypted);

      auto plaintext = crt::decode(decrypted, modulus);
      remainders.push_back(plaintext);
      i++;
    }

    output = decode_crt(remainders, crt, encryption->encoding.isSigned);
  }

  return outcome::success();
}

const std::vector<LweSecretKey> &KeySet::getSecretKeys() const {
  return secretKeys;
}

const std::vector<LweBootstrapKey> &KeySet::getBootstrapKeys() const {
  return bootstrapKeys;
}

const std::vector<LweKeyswitchKey> &KeySet::getKeyswitchKeys() const {
  return keyswitchKeys;
}

const std::vector<PackingKeyswitchKey> &
KeySet::getPackingKeyswitchKeys() const {
  return packingKeyswitchKeys;
}

#ifdef OUTPUT_COMPRESSION_SUPPORT
const std::optional<comp::FullKeys> &KeySet::getFullKey() const {
  return fullKeys;
}
#endif

} // namespace clientlib
} // namespace concretelang

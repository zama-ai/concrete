// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/Support/Error.h"

#define CAPI_ERR_TO_STRINGERROR(instr, msg)                                    \
  {                                                                            \
    int err;                                                                   \
    instr;                                                                     \
    if (err != 0) {                                                            \
      return concretelang::error::StringError(msg);                            \
    }                                                                          \
  }

namespace concretelang {
namespace clientlib {

KeySet::KeySet() : engine(new_engine()) {}

KeySet::~KeySet() {
  for (auto it : secretKeys) {
    free_lwe_secret_key_u64(it.second.second);
  }
  for (auto it : bootstrapKeys) {
    free_lwe_bootstrap_key_u64(it.second.second);
  }
  for (auto it : keyswitchKeys) {
    free_lwe_keyswitch_key_u64(it.second.second);
  }
  free_engine(engine);
}

outcome::checked<std::unique_ptr<KeySet>, StringError>
KeySet::generate(ClientParameters &params, uint64_t seed_msb,
                 uint64_t seed_lsb) {
  auto keySet = std::make_unique<KeySet>();

  OUTCOME_TRYV(keySet->generateKeysFromParams(params, seed_msb, seed_lsb));
  OUTCOME_TRYV(keySet->setupEncryptionMaterial(params, seed_msb, seed_lsb));

  return std::move(keySet);
}

outcome::checked<void, StringError>
KeySet::setupEncryptionMaterial(ClientParameters &params, uint64_t seed_msb,
                                uint64_t seed_lsb) {
  // Set inputs and outputs LWE secret keys
  {
    for (auto param : params.inputs) {
      LweSecretKeyParam secretKeyParam = {0};
      LweSecretKey_u64 *secretKey = nullptr;
      if (param.encryption.hasValue()) {
        auto inputSk = this->secretKeys.find(param.encryption->secretKeyID);
        if (inputSk == this->secretKeys.end()) {
          return StringError("input encryption secret key (")
                 << param.encryption->secretKeyID << ") does not exist ";
        }
        secretKeyParam = inputSk->second.first;
        secretKey = inputSk->second.second;
      }
      std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey_u64 *> input = {
          param, secretKeyParam, secretKey};
      this->inputs.push_back(input);
    }
    for (auto param : params.outputs) {
      LweSecretKeyParam secretKeyParam = {0};
      LweSecretKey_u64 *secretKey = nullptr;
      if (param.encryption.hasValue()) {
        auto outputSk = this->secretKeys.find(param.encryption->secretKeyID);
        if (outputSk == this->secretKeys.end()) {
          return StringError(
              "cannot find output key to generate bootstrap key");
        }
        secretKeyParam = outputSk->second.first;
        secretKey = outputSk->second.second;
      }
      std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey_u64 *> output = {
          param, secretKeyParam, secretKey};
      this->outputs.push_back(output);
    }
  }

  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generateKeysFromParams(ClientParameters &params, uint64_t seed_msb,
                               uint64_t seed_lsb) {

  {
    // Generate LWE secret keys
    for (auto secretKeyParam : params.secretKeys) {
      OUTCOME_TRYV(
          this->generateSecretKey(secretKeyParam.first, secretKeyParam.second));
    }
  }
  // Generate bootstrap and keyswitch keys
  {
    for (auto bootstrapKeyParam : params.bootstrapKeys) {
      OUTCOME_TRYV(this->generateBootstrapKey(bootstrapKeyParam.first,
                                              bootstrapKeyParam.second));
    }
    for (auto keyswitchParam : params.keyswitchKeys) {
      OUTCOME_TRYV(this->generateKeyswitchKey(keyswitchParam.first,
                                              keyswitchParam.second));
    }
  }
  return outcome::success();
}

void KeySet::setKeys(
    std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
        secretKeys,
    std::map<LweSecretKeyID,
             std::pair<BootstrapKeyParam, LweBootstrapKey_u64 *>>
        bootstrapKeys,
    std::map<LweSecretKeyID,
             std::pair<KeyswitchKeyParam, LweKeyswitchKey_u64 *>>
        keyswitchKeys) {
  this->secretKeys = secretKeys;
  this->bootstrapKeys = bootstrapKeys;
  this->keyswitchKeys = keyswitchKeys;
}

outcome::checked<void, StringError>
KeySet::generateSecretKey(LweSecretKeyID id, LweSecretKeyParam param) {
  LweSecretKey_u64 *sk;
  sk = generate_lwe_secret_key_u64(engine, param.dimension);

  secretKeys[id] = {param, sk};

  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generateBootstrapKey(BootstrapKeyID id, BootstrapKeyParam param) {
  // Finding input and output secretKeys
  auto inputSk = secretKeys.find(param.inputSecretKeyID);
  if (inputSk == secretKeys.end()) {
    return StringError("cannot find input key to generate bootstrap key");
  }
  auto outputSk = secretKeys.find(param.outputSecretKeyID);
  if (outputSk == secretKeys.end()) {
    return StringError("cannot find output key to generate bootstrap key");
  }
  // Allocate the bootstrap key
  LweBootstrapKey_u64 *bsk;

  uint64_t total_dimension = outputSk->second.first.dimension;

  assert(total_dimension % param.glweDimension == 0);

  uint64_t polynomialSize = total_dimension / param.glweDimension;

  bsk = generate_lwe_bootstrap_key_u64(
      engine, inputSk->second.second, outputSk->second.second, param.baseLog,
      param.level, param.variance, param.glweDimension, polynomialSize);

  // Store the bootstrap key
  bootstrapKeys[id] = {param, bsk};

  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generateKeyswitchKey(KeyswitchKeyID id, KeyswitchKeyParam param) {
  // Finding input and output secretKeys
  auto inputSk = secretKeys.find(param.inputSecretKeyID);
  if (inputSk == secretKeys.end()) {
    return StringError("cannot find input key to generate keyswitch key");
  }
  auto outputSk = secretKeys.find(param.outputSecretKeyID);
  if (outputSk == secretKeys.end()) {
    return StringError("cannot find output key to generate keyswitch key");
  }
  // Allocate the keyswitch key
  LweKeyswitchKey_u64 *ksk;

  ksk = generate_lwe_keyswitch_key_u64(engine, inputSk->second.second,
                                       outputSk->second.second, param.level,
                                       param.baseLog, param.variance);

  // Store the keyswitch key
  keyswitchKeys[id] = {param, ksk};

  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::allocate_lwe(size_t argPos, uint64_t **ciphertext, uint64_t &size) {
  if (argPos >= inputs.size()) {
    return StringError("allocate_lwe position of argument is too high");
  }
  auto inputSk = inputs[argPos];

  size = std::get<1>(inputSk).lweSize();
  *ciphertext = (uint64_t *)malloc(sizeof(uint64_t) * size);
  return outcome::success();
}

bool KeySet::isInputEncrypted(size_t argPos) {
  return argPos < inputs.size() &&
         std::get<0>(inputs[argPos]).encryption.hasValue();
}

bool KeySet::isOutputEncrypted(size_t argPos) {
  return argPos < outputs.size() &&
         std::get<0>(outputs[argPos]).encryption.hasValue();
}

outcome::checked<void, StringError>
KeySet::encrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t input) {
  if (argPos >= inputs.size()) {
    return StringError("encrypt_lwe position of argument is too high");
  }
  auto inputSk = inputs[argPos];
  if (!std::get<0>(inputSk).encryption.hasValue()) {
    return StringError("encrypt_lwe the positional argument is not encrypted");
  }
  // Encode - TODO we could check if the input value is in the right range
  uint64_t plaintext =
      input << (64 - (std::get<0>(inputSk).encryption->encoding.precision + 1));
  ::encrypt_lwe_u64(engine, std::get<2>(inputSk), ciphertext, plaintext,
                    std::get<0>(inputSk).encryption->variance);
  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::decrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t &output) {

  if (argPos >= outputs.size()) {
    return StringError("decrypt_lwe: position of argument is too high");
  }
  auto outputSk = outputs[argPos];
  if (!std::get<0>(outputSk).encryption.hasValue()) {
    return StringError("decrypt_lwe: the positional argument is not encrypted");
  }
  uint64_t plaintext =
      ::decrypt_lwe_u64(engine, std::get<2>(outputSk), ciphertext);
  // Decode
  size_t precision = std::get<0>(outputSk).encryption->encoding.precision;
  output = plaintext >> (64 - precision - 2);
  size_t carry = output % 2;
  output = ((output >> 1) + carry) % (1 << (precision + 1));
  return outcome::success();
}

const std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey_u64 *>>
    &KeySet::getSecretKeys() {
  return secretKeys;
}

const std::map<LweSecretKeyID,
               std::pair<BootstrapKeyParam, LweBootstrapKey_u64 *>> &
KeySet::getBootstrapKeys() {
  return bootstrapKeys;
}

const std::map<LweSecretKeyID,
               std::pair<KeyswitchKeyParam, LweKeyswitchKey_u64 *>> &
KeySet::getKeyswitchKeys() {
  return keyswitchKeys;
}

} // namespace clientlib
} // namespace concretelang

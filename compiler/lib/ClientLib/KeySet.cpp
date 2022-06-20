// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/CRT.h"
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
  _clientParameters = params;

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
             std::pair<BootstrapKeyParam, std::shared_ptr<LweBootstrapKey>>>
        bootstrapKeys,
    std::map<LweSecretKeyID,
             std::pair<KeyswitchKeyParam, std::shared_ptr<LweKeyswitchKey>>>
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
  bootstrapKeys[id] = {param, std::make_shared<LweBootstrapKey>(bsk)};

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
  keyswitchKeys[id] = {param, std::make_shared<LweKeyswitchKey>(ksk)};

  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::allocate_lwe(size_t argPos, uint64_t **ciphertext, uint64_t &size) {
  if (argPos >= inputs.size()) {
    return StringError("allocate_lwe position of argument is too high");
  }
  auto inputSk = inputs[argPos];
  auto encryption = std::get<0>(inputSk).encryption;
  if (!encryption.hasValue()) {
    return StringError("allocate_lwe argument #")
           << argPos << "is not encypeted";
  }
  auto numBlocks =
      encryption->encoding.crt.empty() ? 1 : encryption->encoding.crt.size();

  size = std::get<1>(inputSk).lweSize();
  *ciphertext = (uint64_t *)malloc(sizeof(uint64_t) * size * numBlocks);
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

/// Return the number of bits to represents the given value
uint64_t bitWidthOfValue(uint64_t value) { return std::ceil(std::log2(value)); }

outcome::checked<void, StringError>
KeySet::encrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t input) {
  if (argPos >= inputs.size()) {
    return StringError("encrypt_lwe position of argument is too high");
  }
  auto inputSk = inputs[argPos];
  auto encryption = std::get<0>(inputSk).encryption;
  if (!encryption.hasValue()) {
    return StringError("encrypt_lwe the positional argument is not encrypted");
  }
  auto encoding = encryption->encoding;
  auto lweSecretKeyParam = std::get<1>(inputSk);
  auto lweSecretKey = std::get<2>(inputSk);
  // CRT encoding - N blocks with crt encoding
  auto crt = encryption->encoding.crt;
  if (!crt.empty()) {
    // Put each decomposition into a new ciphertext
    auto product = crt::productOfModuli(crt);
    for (auto modulus : crt) {
      auto plaintext = crt::encode(input, modulus, product);
      ::encrypt_lwe_u64(engine, lweSecretKey, ciphertext, plaintext,
                        encryption->variance);
      ciphertext = ciphertext + lweSecretKeyParam.lweSize();
    }
    return outcome::success();
  }
  // Simple TFHE integers - 1 blocks with one padding bits
  // TODO we could check if the input value is in the right range
  uint64_t plaintext = input << (64 - (encryption->encoding.precision + 1));
  ::encrypt_lwe_u64(engine, lweSecretKey, ciphertext, plaintext,
                    encryption->variance);
  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::decrypt_lwe(size_t argPos, uint64_t *ciphertext, uint64_t &output) {
  if (argPos >= outputs.size()) {
    return StringError("decrypt_lwe: position of argument is too high");
  }
  auto outputSk = outputs[argPos];
  auto lweSecretKey = std::get<2>(outputSk);
  auto lweSecretKeyParam = std::get<1>(outputSk);
  auto encryption = std::get<0>(outputSk).encryption;
  if (!encryption.hasValue()) {
    return StringError("decrypt_lwe: the positional argument is not encrypted");
  }
  auto crt = encryption->encoding.crt;
  // CRT encoding - N blocks with crt encoding
  if (!crt.empty()) {
    std::vector<int64_t> remainders;
    // decrypt and decode remainders
    for (auto modulus : crt) {
      auto decrypted = ::decrypt_lwe_u64(engine, lweSecretKey, ciphertext);
      auto plaintext = crt::decode(decrypted, modulus);
      remainders.push_back(plaintext);
      ciphertext = ciphertext + lweSecretKeyParam.lweSize();
    }
    // compute the inverse crt
    output = crt::iCrt(crt, remainders);
    return outcome::success();
  }
  // Simple TFHE integers - 1 blocks with one padding bits
  uint64_t plaintext = ::decrypt_lwe_u64(engine, lweSecretKey, ciphertext);
  // Decode
  size_t precision = encryption->encoding.precision;
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
               std::pair<BootstrapKeyParam, std::shared_ptr<LweBootstrapKey>>> &
KeySet::getBootstrapKeys() {
  return bootstrapKeys;
}

const std::map<LweSecretKeyID,
               std::pair<KeyswitchKeyParam, std::shared_ptr<LweKeyswitchKey>>> &
KeySet::getKeyswitchKeys() {
  return keyswitchKeys;
}

} // namespace clientlib
} // namespace concretelang

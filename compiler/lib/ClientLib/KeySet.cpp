// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/ClientLib/CRT.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Runtime/seeder.h"
#include "concretelang/Support/Error.h"

#define CAPI_ERR_TO_STRINGERROR(instr, msg)                                    \
  {                                                                            \
    int err;                                                                   \
    instr;                                                                     \
    if (err != 0) {                                                            \
      return concretelang::error::StringError(msg);                            \
    }                                                                          \
  }

int clone_transform_lwe_secret_key_to_glwe_secret_key_u64(
    DefaultEngine *default_engine, LweSecretKey64 *output_lwe_sk,
    size_t poly_size, GlweSecretKey64 **output_glwe_sk) {
  LweSecretKey64 *output_lwe_sk_clone = NULL;
  int lwe_out_sk_clone_ok =
      clone_lwe_secret_key_u64(output_lwe_sk, &output_lwe_sk_clone);
  if (lwe_out_sk_clone_ok != 0) {
    return 1;
  }

  int glwe_sk_ok =
      default_engine_transform_lwe_secret_key_to_glwe_secret_key_u64(
          default_engine, &output_lwe_sk_clone, poly_size, output_glwe_sk);
  if (glwe_sk_ok != 0) {
    return 1;
  }

  if (output_lwe_sk_clone != NULL) {
    return 1;
  }

  return 0;
}

namespace concretelang {
namespace clientlib {

KeySet::KeySet() {

  CAPI_ASSERT_ERROR(new_default_engine(best_seeder, &engine));

  CAPI_ASSERT_ERROR(new_default_parallel_engine(best_seeder, &par_engine));
}

KeySet::~KeySet() {
  for (auto it : secretKeys) {
    CAPI_ASSERT_ERROR(destroy_lwe_secret_key_u64(it.second.second));
  }

  CAPI_ASSERT_ERROR(destroy_default_engine(engine));
  CAPI_ASSERT_ERROR(destroy_default_parallel_engine(par_engine));
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
      LweSecretKey64 *secretKey = nullptr;
      if (param.encryption.hasValue()) {
        auto inputSk = this->secretKeys.find(param.encryption->secretKeyID);
        if (inputSk == this->secretKeys.end()) {
          return StringError("input encryption secret key (")
                 << param.encryption->secretKeyID << ") does not exist ";
        }
        secretKeyParam = inputSk->second.first;
        secretKey = inputSk->second.second;
      }
      std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey64 *> input = {
          param, secretKeyParam, secretKey};
      this->inputs.push_back(input);
    }
    for (auto param : params.outputs) {
      LweSecretKeyParam secretKeyParam = {0};
      LweSecretKey64 *secretKey = nullptr;
      if (param.encryption.hasValue()) {
        auto outputSk = this->secretKeys.find(param.encryption->secretKeyID);
        if (outputSk == this->secretKeys.end()) {
          return StringError(
              "cannot find output key to generate bootstrap key");
        }
        secretKeyParam = outputSk->second.first;
        secretKey = outputSk->second.second;
      }
      std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey64 *> output = {
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
  // Generate bootstrap, keyswitch and packing keyswitch keys
  {
    for (auto bootstrapKeyParam : params.bootstrapKeys) {
      OUTCOME_TRYV(this->generateBootstrapKey(bootstrapKeyParam.first,
                                              bootstrapKeyParam.second));
    }
    for (auto keyswitchParam : params.keyswitchKeys) {
      OUTCOME_TRYV(this->generateKeyswitchKey(keyswitchParam.first,
                                              keyswitchParam.second));
    }
    for (auto packingParam : params.packingKeys) {
      OUTCOME_TRYV(
          this->generatePackingKey(packingParam.first, packingParam.second));
    }
  }
  return outcome::success();
}

void KeySet::setKeys(
    std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey64 *>>
        secretKeys,
    std::map<LweSecretKeyID,
             std::pair<BootstrapKeyParam, std::shared_ptr<LweBootstrapKey>>>
        bootstrapKeys,
    std::map<LweSecretKeyID,
             std::pair<KeyswitchKeyParam, std::shared_ptr<LweKeyswitchKey>>>
        keyswitchKeys,
    std::map<LweSecretKeyID, std::pair<PackingKeySwitchParam,
                                       std::shared_ptr<PackingKeyswitchKey>>>
        packingKeys) {
  this->secretKeys = secretKeys;
  this->bootstrapKeys = bootstrapKeys;
  this->keyswitchKeys = keyswitchKeys;
  this->packingKeys = packingKeys;
}

outcome::checked<void, StringError>
KeySet::generateSecretKey(LweSecretKeyID id, LweSecretKeyParam param) {
  LweSecretKey64 *sk;
  CAPI_ASSERT_ERROR(default_engine_generate_new_lwe_secret_key_u64(
      engine, param.dimension, &sk));

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
  LweBootstrapKey64 *bsk;

  uint64_t total_dimension = outputSk->second.first.dimension;

  assert(total_dimension % param.glweDimension == 0);

  uint64_t polynomialSize = total_dimension / param.glweDimension;

  GlweSecretKey64 *output_glwe_sk = nullptr;

  // This is not part of the C FFI but rather is a C util exposed for
  // convenience in tests.
  CAPI_ASSERT_ERROR(clone_transform_lwe_secret_key_to_glwe_secret_key_u64(
      engine, outputSk->second.second, polynomialSize, &output_glwe_sk));

  CAPI_ASSERT_ERROR(default_parallel_engine_generate_new_lwe_bootstrap_key_u64(
      par_engine, inputSk->second.second, output_glwe_sk, param.baseLog,
      param.level, param.variance, &bsk));

  CAPI_ASSERT_ERROR(destroy_glwe_secret_key_u64(output_glwe_sk));

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
  LweKeyswitchKey64 *ksk;

  CAPI_ASSERT_ERROR(default_engine_generate_new_lwe_keyswitch_key_u64(
      engine, inputSk->second.second, outputSk->second.second, param.level,
      param.baseLog, param.variance, &ksk));

  // Store the keyswitch key
  keyswitchKeys[id] = {param, std::make_shared<LweKeyswitchKey>(ksk)};

  return outcome::success();
}

outcome::checked<void, StringError>
KeySet::generatePackingKey(PackingKeySwitchID id, PackingKeySwitchParam param) {
  // Finding input secretKeys
  auto inputSk = secretKeys.find(param.inputSecretKeyID);
  if (inputSk == secretKeys.end()) {
    return StringError(
        "cannot find input key to generate packing keyswitch key");
  }
  auto bsk = bootstrapKeys.find(param.bootstrapKeyID);
  if (bsk == bootstrapKeys.end()) {
    return StringError(
        "cannot find input key to generate packing keyswitch key");
  }

  // This is not part of the C FFI but rather is a C util exposed for
  // convenience in tests.
  GlweSecretKey64 *output_glwe_sk = nullptr;

  auto lweDimension =
      inputSk->second.first.lweDimension() / bsk->second.first.glweDimension;

  CAPI_ASSERT_ERROR(clone_transform_lwe_secret_key_to_glwe_secret_key_u64(
      engine, inputSk->second.second, lweDimension, &output_glwe_sk));

  // Allocate the packing keyswitch key
  LweCircuitBootstrapPrivateFunctionalPackingKeyswitchKeys64 *fpksk;
  CAPI_ASSERT_ERROR(
      default_parallel_engine_generate_new_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_unchecked_u64(
          par_engine, inputSk->second.second, output_glwe_sk, param.baseLog,
          param.level, param.variance, &fpksk));

  // Store the keyswitch key
  packingKeys[id] = {param, std::make_shared<PackingKeyswitchKey>(fpksk)};

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
      CAPI_ASSERT_ERROR(
          default_engine_discard_encrypt_lwe_ciphertext_u64_raw_ptr_buffers(
              engine, lweSecretKey, ciphertext, plaintext,
              encryption->variance));

      ciphertext = ciphertext + lweSecretKeyParam.lweSize();
    }
    return outcome::success();
  }
  // Simple TFHE integers - 1 blocks with one padding bits
  // TODO we could check if the input value is in the right range
  uint64_t plaintext = input << (64 - (encryption->encoding.precision + 1));
  CAPI_ASSERT_ERROR(
      default_engine_discard_encrypt_lwe_ciphertext_u64_raw_ptr_buffers(
          engine, lweSecretKey, ciphertext, plaintext, encryption->variance));

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
      uint64_t decrypted;
      CAPI_ASSERT_ERROR(
          default_engine_decrypt_lwe_ciphertext_u64_raw_ptr_buffers(
              engine, lweSecretKey, ciphertext, &decrypted));

      auto plaintext = crt::decode(decrypted, modulus);
      remainders.push_back(plaintext);
      ciphertext = ciphertext + lweSecretKeyParam.lweSize();
    }
    // compute the inverse crt
    output = crt::iCrt(crt, remainders);
    return outcome::success();
  }
  // Simple TFHE integers - 1 blocks with one padding bits
  uint64_t plaintext;

  CAPI_ASSERT_ERROR(default_engine_decrypt_lwe_ciphertext_u64_raw_ptr_buffers(
      engine, lweSecretKey, ciphertext, &plaintext));

  // Decode
  uint64_t precision = encryption->encoding.precision;
  output = plaintext >> (64 - precision - 2);
  auto carry = output % 2;
  uint64_t mod = (((uint64_t)1) << (precision + 1));
  output = ((output >> 1) + carry) % mod;

  return outcome::success();
}

const std::map<LweSecretKeyID, std::pair<LweSecretKeyParam, LweSecretKey64 *>> &
KeySet::getSecretKeys() {
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

const std::map<LweSecretKeyID, std::pair<PackingKeySwitchParam,
                                         std::shared_ptr<PackingKeyswitchKey>>>
    &KeySet::getPackingKeys() {
  return packingKeys;
}

} // namespace clientlib
} // namespace concretelang

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/Support/Error.h"

#define CAPI_ERR_TO_LLVM_ERROR(s, msg)                                         \
  {                                                                            \
    int err;                                                                   \
    s;                                                                         \
    if (err != 0) {                                                            \
      return llvm::make_error<llvm::StringError>(                              \
          msg, llvm::inconvertibleErrorCode());                                \
    }                                                                          \
  }

namespace mlir {
namespace concretelang {

KeySet::~KeySet() {
  int err;
  for (auto it : secretKeys) {
    free_lwe_secret_key_u64(&err, it.second.second);
  }
  for (auto it : bootstrapKeys) {
    free_lwe_bootstrap_key_u64(&err, it.second.second);
  }
  for (auto it : keyswitchKeys) {
    free_lwe_keyswitch_key_u64(&err, it.second.second);
  }
  free_encryption_generator(&err, encryptionRandomGenerator);
}

llvm::Expected<std::unique_ptr<KeySet>>
KeySet::generate(ClientParameters &params, uint64_t seed_msb,
                 uint64_t seed_lsb) {

  auto keySet = uninitialized();

  if (auto error = keySet->generateKeysFromParams(params, seed_msb, seed_lsb)) {
    return std::move(error);
  }

  if (auto error =
          keySet->setupEncryptionMaterial(params, seed_msb, seed_lsb)) {
    return std::move(error);
  }

  return std::move(keySet);
}

std::unique_ptr<KeySet> KeySet::uninitialized() {
  return std::make_unique<KeySet>();
}

llvm::Error KeySet::setupEncryptionMaterial(ClientParameters &params,
                                            uint64_t seed_msb,
                                            uint64_t seed_lsb) {
  // Set inputs and outputs LWE secret keys
  {
    for (auto param : params.inputs) {
      LweSecretKeyParam secretKeyParam = {0};
      LweSecretKey_u64 *secretKey = nullptr;
      if (param.encryption.hasValue()) {
        auto inputSk = this->secretKeys.find(param.encryption->secretKeyID);
        if (inputSk == this->secretKeys.end()) {
          return llvm::make_error<llvm::StringError>(
              "input encryption secret key (" + param.encryption->secretKeyID +
                  ") does not exist ",
              llvm::inconvertibleErrorCode());
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
          return llvm::make_error<llvm::StringError>(
              "cannot find output key to generate bootstrap key",
              llvm::inconvertibleErrorCode());
        }
        secretKeyParam = outputSk->second.first;
        secretKey = outputSk->second.second;
      }
      std::tuple<CircuitGate, LweSecretKeyParam, LweSecretKey_u64 *> output = {
          param, secretKeyParam, secretKey};
      this->outputs.push_back(output);
    }
  }
  int err;
  CAPI_ERR_TO_LLVM_ERROR(
      this->encryptionRandomGenerator =
          allocate_encryption_generator(&err, seed_msb, seed_lsb),
      "cannot allocate encryption generator");

  return llvm::Error::success();
}

llvm::Error KeySet::generateKeysFromParams(ClientParameters &params,
                                           uint64_t seed_msb,
                                           uint64_t seed_lsb) {

  {
    // Generate LWE secret keys
    SecretRandomGenerator *generator;
    CAPI_ERR_TO_LLVM_ERROR(
        generator = allocate_secret_generator(&err, seed_msb, seed_lsb),
        "cannot allocate random generator");
    for (auto secretKeyParam : params.secretKeys) {
      auto e = this->generateSecretKey(secretKeyParam.first,
                                       secretKeyParam.second, generator);
      if (e) {
        return std::move(e);
      }
    }
    CAPI_ERR_TO_LLVM_ERROR(free_secret_generator(&err, generator),
                           "cannot free random generator");
  }
  // Allocate the encryption random generator
  CAPI_ERR_TO_LLVM_ERROR(
      this->encryptionRandomGenerator =
          allocate_encryption_generator(&err, seed_msb, seed_lsb),
      "cannot allocate encryption generator");
  // Generate bootstrap and keyswitch keys
  {
    for (auto bootstrapKeyParam : params.bootstrapKeys) {
      auto e = this->generateBootstrapKey(bootstrapKeyParam.first,
                                          bootstrapKeyParam.second,
                                          this->encryptionRandomGenerator);
      if (e) {
        return std::move(e);
      }
    }
    for (auto keyswitchParam : params.keyswitchKeys) {
      auto e = this->generateKeyswitchKey(keyswitchParam.first,
                                          keyswitchParam.second,
                                          this->encryptionRandomGenerator);
      if (e) {
        return std::move(e);
      }
    }
  }
  return llvm::Error::success();
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

llvm::Error KeySet::generateSecretKey(LweSecretKeyID id,
                                      LweSecretKeyParam param,
                                      SecretRandomGenerator *generator) {
  LweSecretKey_u64 *sk;
  CAPI_ERR_TO_LLVM_ERROR(
      sk = allocate_lwe_secret_key_u64(&err, {param.size + 1}),
      "cannot allocate secret key");

  CAPI_ERR_TO_LLVM_ERROR(fill_lwe_secret_key_u64(&err, sk, generator),
                         "cannot fill secret key with random generator");

  secretKeys[id] = {param, sk};

  return llvm::Error::success();
}

llvm::Error KeySet::generateBootstrapKey(BootstrapKeyID id,
                                         BootstrapKeyParam param,
                                         EncryptionRandomGenerator *generator) {
  // Finding input and output secretKeys
  auto inputSk = secretKeys.find(param.inputSecretKeyID);
  if (inputSk == secretKeys.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find input key to generate bootstrap key",
        llvm::inconvertibleErrorCode());
  }
  auto outputSk = secretKeys.find(param.outputSecretKeyID);
  if (outputSk == secretKeys.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find output key to generate bootstrap key",
        llvm::inconvertibleErrorCode());
  }
  // Allocate the bootstrap key
  LweBootstrapKey_u64 *bsk;

  uint64_t total_dimension = outputSk->second.first.size;

  assert(total_dimension % param.glweDimension == 0);

  uint64_t polynomialSize = total_dimension / param.glweDimension;

  CAPI_ERR_TO_LLVM_ERROR(
      bsk = allocate_lwe_bootstrap_key_u64(
          &err, {param.level}, {param.baseLog}, {param.glweDimension + 1},
          {inputSk->second.first.size + 1}, {polynomialSize}),
      "cannot allocate bootstrap key");

  // Store the bootstrap key
  bootstrapKeys[id] = {param, bsk};

  // Convert the output lwe key to glwe key
  GlweSecretKey_u64 *glwe_sk;

  CAPI_ERR_TO_LLVM_ERROR(
      glwe_sk = allocate_glwe_secret_key_u64(&err, {param.glweDimension + 1},
                                             {polynomialSize}),
      "cannot allocate glwe key for initiliazation of bootstrap key");

  CAPI_ERR_TO_LLVM_ERROR(fill_glwe_secret_key_with_lwe_secret_key_u64(
                             &err, glwe_sk, outputSk->second.second),
                         "cannot fill glwe key with big key");

  // Initialize the bootstrap key
  CAPI_ERR_TO_LLVM_ERROR(
      fill_lwe_bootstrap_key_u64(&err, bsk, inputSk->second.second, glwe_sk,
                                 generator, {param.variance}),
      "cannot fill bootstrap key");
  CAPI_ERR_TO_LLVM_ERROR(
      free_glwe_secret_key_u64(&err, glwe_sk),
      "cannot free glwe key for initiliazation of bootstrap key")
  return llvm::Error::success();
}

llvm::Error KeySet::generateKeyswitchKey(KeyswitchKeyID id,
                                         KeyswitchKeyParam param,
                                         EncryptionRandomGenerator *generator) {
  // Finding input and output secretKeys
  auto inputSk = secretKeys.find(param.inputSecretKeyID);
  if (inputSk == secretKeys.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find input key to generate keyswitch key",
        llvm::inconvertibleErrorCode());
  }
  auto outputSk = secretKeys.find(param.outputSecretKeyID);
  if (outputSk == secretKeys.end()) {
    return llvm::make_error<llvm::StringError>(
        "cannot find input key to generate keyswitch key",
        llvm::inconvertibleErrorCode());
  }
  // Allocate the keyswitch key
  LweKeyswitchKey_u64 *ksk;
  CAPI_ERR_TO_LLVM_ERROR(
      ksk = allocate_lwe_keyswitch_key_u64(&err, {param.level}, {param.baseLog},
                                           {inputSk->second.first.size + 1},
                                           {outputSk->second.first.size + 1}),
      "cannot allocate keyswitch key");
  // Store the keyswitch key
  keyswitchKeys[id] = {param, ksk};
  // Initialize the keyswitch key
  CAPI_ERR_TO_LLVM_ERROR(
      fill_lwe_keyswitch_key_u64(&err, ksk, inputSk->second.second,
                                 outputSk->second.second, generator,
                                 {param.variance}),
      "cannot fill bootsrap key");
  return llvm::Error::success();
}

llvm::Error KeySet::allocate_lwe(size_t argPos,
                                 LweCiphertext_u64 **ciphertext) {
  if (argPos >= inputs.size()) {
    return llvm::make_error<llvm::StringError>(
        "allocate_lwe position of argument is too high",
        llvm::inconvertibleErrorCode());
  }
  auto inputSk = inputs[argPos];
  CAPI_ERR_TO_LLVM_ERROR(*ciphertext = allocate_lwe_ciphertext_u64(
                             &err, {std::get<1>(inputSk).size + 1}),
                         "cannot allocate ciphertext");
  return llvm::Error::success();
}

bool KeySet::isInputEncrypted(size_t argPos) {
  return argPos < inputs.size() &&
         std::get<0>(inputs[argPos]).encryption.hasValue();
}

bool KeySet::isOutputEncrypted(size_t argPos) {
  return argPos < outputs.size() &&
         std::get<0>(outputs[argPos]).encryption.hasValue();
}

llvm::Error KeySet::encrypt_lwe(size_t argPos, LweCiphertext_u64 *ciphertext,
                                uint64_t input) {
  if (argPos >= inputs.size()) {
    return llvm::make_error<llvm::StringError>(
        "encrypt_lwe position of argument is too high",
        llvm::inconvertibleErrorCode());
  }
  auto inputSk = inputs[argPos];
  if (!std::get<0>(inputSk).encryption.hasValue()) {
    return llvm::make_error<llvm::StringError>(
        "encrypt_lwe the positional argument is not encrypted",
        llvm::inconvertibleErrorCode());
  }
  // Encode - TODO we could check if the input value is in the right range
  Plaintext_u64 plaintext = {
      input << (64 -
                (std::get<0>(inputSk).encryption->encoding.precision + 1))};
  // Encrypt
  CAPI_ERR_TO_LLVM_ERROR(
      encrypt_lwe_u64(&err, std::get<2>(inputSk), ciphertext, plaintext,
                      encryptionRandomGenerator,
                      {std::get<0>(inputSk).encryption->variance}),
      "cannot encrypt");
  return llvm::Error::success();
}

llvm::Error KeySet::decrypt_lwe(size_t argPos, LweCiphertext_u64 *ciphertext,
                                uint64_t &output) {

  if (argPos >= outputs.size()) {
    return llvm::make_error<llvm::StringError>(
        "decrypt_lwe: position of argument is too high",
        llvm::inconvertibleErrorCode());
  }
  auto outputSk = outputs[argPos];
  if (!std::get<0>(outputSk).encryption.hasValue()) {
    return llvm::make_error<llvm::StringError>(
        "decrypt_lwe: the positional argument is not encrypted",
        llvm::inconvertibleErrorCode());
  }
  // Decrypt
  Plaintext_u64 plaintext = {0};
  CAPI_ERR_TO_LLVM_ERROR(
      decrypt_lwe_u64(&err, std::get<2>(outputSk), ciphertext, &plaintext),
      "cannot decrypt");
  // Decode
  size_t precision = std::get<0>(outputSk).encryption->encoding.precision;
  output = plaintext._0 >> (64 - precision - 2);
  size_t carry = output % 2;
  output = ((output >> 1) + carry) % (1 << (precision + 1));
  return llvm::Error::success();
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

} // namespace concretelang
} // namespace mlir

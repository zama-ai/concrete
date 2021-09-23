#include "zamalang/Support/KeySet.h"

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
namespace zamalang {

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
  auto keySet = std::make_unique<KeySet>();

  {
    // Generate LWE secret keys
    SecretRandomGenerator *generator;
    CAPI_ERR_TO_LLVM_ERROR(
        generator = allocate_secret_generator(&err, seed_msb, seed_lsb),
        "cannot allocate random generator");
    for (auto secretKeyParam : params.secretKeys) {
      auto e = keySet->generateSecretKey(secretKeyParam.first,
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
      keySet->encryptionRandomGenerator =
          allocate_encryption_generator(&err, seed_msb, seed_lsb),
      "cannot allocate encryption generator");
  // Generate bootstrap and keyswitch keys
  {
    for (auto bootstrapKeyParam : params.bootstrapKeys) {
      auto e = keySet->generateBootstrapKey(bootstrapKeyParam.first,
                                            bootstrapKeyParam.second,
                                            keySet->encryptionRandomGenerator);
      if (e) {
        return std::move(e);
      }
    }
    for (auto keyswitchParam : params.keyswitchKeys) {
      auto e = keySet->generateKeyswitchKey(keyswitchParam.first,
                                            keyswitchParam.second,
                                            keySet->encryptionRandomGenerator);
      if (e) {
        return std::move(e);
      }
    }
  }
  // Set inputs and outputs LWE secret keys
  {
    for (auto param : params.inputs) {
      std::tuple<CircuitGate, LweSecretKeyParam *, LweSecretKey_u64 *> input = {
          param, nullptr, nullptr};
      if (param.encryption.hasValue()) {
        auto inputSk = keySet->secretKeys.find(param.encryption->secretKeyID);
        if (inputSk == keySet->secretKeys.end()) {
          return llvm::make_error<llvm::StringError>(
              "cannot find input key to generate bootstrap key",
              llvm::inconvertibleErrorCode());
        }
        std::get<1>(input) = &inputSk->second.first;
        std::get<2>(input) = inputSk->second.second;
      }
      keySet->inputs.push_back(input);
    }
    for (auto param : params.outputs) {
      std::tuple<CircuitGate, LweSecretKeyParam *, LweSecretKey_u64 *> output =
          {param, nullptr, nullptr};
      if (param.encryption.hasValue()) {
        auto outputSk = keySet->secretKeys.find(param.encryption->secretKeyID);
        if (outputSk == keySet->secretKeys.end()) {
          return llvm::make_error<llvm::StringError>(
              "cannot find output key to generate bootstrap key",
              llvm::inconvertibleErrorCode());
        }
        std::get<1>(output) = &outputSk->second.first;
        std::get<2>(output) = outputSk->second.second;
      }
      keySet->outputs.push_back(output);
    }
  }
  return std::move(keySet);
}

llvm::Error KeySet::generateSecretKey(LweSecretKeyID id,
                                      LweSecretKeyParam param,
                                      SecretRandomGenerator *generator) {
  LweSecretKey_u64 *sk;
  CAPI_ERR_TO_LLVM_ERROR(sk = allocate_lwe_secret_key_u64(&err, {param.size}),
                         "cannot allocate secret key");
  CAPI_ERR_TO_LLVM_ERROR(fill_lwe_secret_key_u64(&err, sk, generator),
                         "cannot fill secret key with random generator")
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
        "cannot find input key to generate bootstrap key",
        llvm::inconvertibleErrorCode());
  }
  // Allocate the bootstrap key
  LweBootstrapKey_u64 *bsk;
  CAPI_ERR_TO_LLVM_ERROR(
      bsk = allocate_lwe_bootstrap_key_u64(
          &err, {param.level}, {param.baseLog}, {param.k},
          {inputSk->second.first.size},
          {outputSk->second.first.size /*TODO: size / k ?*/}),
      "cannot allocate bootstrap key");
  // Store the bootstrap key
  bootstrapKeys[id] = {param, bsk};
  // Convert the output lwe key to glwe key
  GlweSecretKey_u64 *glwe_sk;
  CAPI_ERR_TO_LLVM_ERROR(
      glwe_sk = allocate_glwe_secret_key_u64(&err, {param.k},
                                             {outputSk->second.first.size}),
      "cannot allocate glwe key for initiliazation of bootstrap key");
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
                                           {inputSk->second.first.size},
                                           {outputSk->second.first.size}),
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
                             &err, {std::get<1>(inputSk)->size}),
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

} // namespace zamalang
} // namespace mlir
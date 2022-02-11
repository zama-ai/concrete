// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/ClientLib/KeySet.h"
#include "concretelang/Support/Error.h"

namespace mlir {
namespace concretelang {

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
  free_encryption_generator(encryptionRandomGenerator);
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

  this->encryptionRandomGenerator =
      allocate_encryption_generator(seed_msb, seed_lsb);

  return llvm::Error::success();
}

llvm::Error KeySet::generateKeysFromParams(ClientParameters &params,
                                           uint64_t seed_msb,
                                           uint64_t seed_lsb) {
  {
    // Generate LWE secret keys
    SecretRandomGenerator *generator;

    generator = allocate_secret_generator(seed_msb, seed_lsb);
    for (auto secretKeyParam : params.secretKeys) {
      auto e = this->generateSecretKey(secretKeyParam.first,
                                       secretKeyParam.second, generator);
      if (e) {
        return std::move(e);
      }
    }
    free_secret_generator(generator);
  }
  // Allocate the encryption random generator

  this->encryptionRandomGenerator =
      allocate_encryption_generator(seed_msb, seed_lsb);
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
  sk = allocate_lwe_secret_key_u64({param.size});

  fill_lwe_secret_key_u64(sk, generator);

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

  bsk = allocate_lwe_bootstrap_key_u64(
      {param.level}, {param.baseLog}, {param.glweDimension},
      {inputSk->second.first.size}, {polynomialSize});

  // Store the bootstrap key
  bootstrapKeys[id] = {param, bsk};

  // Convert the output lwe key to glwe key
  GlweSecretKey_u64 *glwe_sk;

  glwe_sk =
      allocate_glwe_secret_key_u64({param.glweDimension}, {polynomialSize});

  fill_glwe_secret_key_with_lwe_secret_key_u64(glwe_sk,
                                               outputSk->second.second);

  // Initialize the bootstrap key
  fill_lwe_bootstrap_key_u64(bsk, inputSk->second.second, glwe_sk, generator,
                             {param.variance});
  free_glwe_secret_key_u64(glwe_sk);
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

  ksk = allocate_lwe_keyswitch_key_u64({param.level}, {param.baseLog},
                                       {inputSk->second.first.size},
                                       {outputSk->second.first.size});

  // Store the keyswitch key
  keyswitchKeys[id] = {param, ksk};
  // Initialize the keyswitch key

  fill_lwe_keyswitch_key_u64(ksk, inputSk->second.second,
                             outputSk->second.second, generator,
                             {param.variance});
  return llvm::Error::success();
}

llvm::Error KeySet::allocate_lwe(size_t argPos, uint64_t **ciphertext,
                                 uint64_t &size) {
  if (argPos >= inputs.size()) {
    return llvm::make_error<llvm::StringError>(
        "allocate_lwe position of argument is too high",
        llvm::inconvertibleErrorCode());
  }
  auto inputSk = inputs[argPos];

  size = std::get<1>(inputSk).size + 1;
  *ciphertext = (uint64_t *)malloc(sizeof(uint64_t) * size);
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

llvm::Error KeySet::encrypt_lwe(size_t argPos, uint64_t *ciphertext,
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
  uint64_t plaintext =
      input << (64 - (std::get<0>(inputSk).encryption->encoding.precision + 1));
  encrypt_lwe_u64(std::get<2>(inputSk), ciphertext, plaintext,
                  encryptionRandomGenerator,
                  {std::get<0>(inputSk).encryption->variance});
  return llvm::Error::success();
}

llvm::Error KeySet::decrypt_lwe(size_t argPos, uint64_t *ciphertext,
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
  uint64_t plaintext = decrypt_lwe_u64(std::get<2>(outputSk), ciphertext);
  // Decode
  size_t precision = std::get<0>(outputSk).encryption->encoding.precision;
  output = plaintext >> (64 - precision - 2);
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

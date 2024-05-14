
// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Common/Keys.h"
#include "capnp/any.h"
#include "concrete-cpu.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Protocol.h"
#include <climits>
#include <cstdint>
#include <memory>
#include <stdlib.h>

using concretelang::csprng::EncryptionCSPRNG;
using concretelang::csprng::SecretCSPRNG;
using concretelang::protocol::Message;
using concretelang::protocol::protoPayloadToSharedVector;
using concretelang::protocol::vectorToProtoPayload;

namespace concretelang {
namespace keys {

template <typename ProtoKey, typename ProtoKeyInfo, typename Key>
Message<ProtoKey> keyToProto(const Key &key) {
  Message<ProtoKey> output;
  auto proto = output.asBuilder();
  proto.setInfo(key.getInfo().asReader());
  proto.setPayload(vectorToProtoPayload(key.getTransportBuffer()).asReader());
  return std::move(output);
}

void writeSeed(struct Uint128 seed, std::vector<uint64_t> &buffer) {
  csprng::writeSeed(seed, buffer.data());
}

void readSeed(struct Uint128 &seed, std::vector<uint64_t> &buffer) {
  csprng::readSeed(seed, buffer.data());
}

LweSecretKey::LweSecretKey(Message<concreteprotocol::LweSecretKeyInfo> info,
                           SecretCSPRNG &csprng)
    : LweSecretKey(info) {
  // Allocate the buffer
  buffer = std::make_shared<std::vector<uint64_t>>(
      info.asReader().getParams().getLweDimension());

  // We copy the information.
  this->info = info;

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  // In insecure debug mode, the secret key is filled with zeros.
  getApproval();
  std::fill(buffer->begin(), buffer->end(), 0);
#else
  // Initialize the lwe secret key buffer
  concrete_cpu_init_secret_key_u64(
      buffer->data(), info.asReader().getParams().getLweDimension(),
      csprng.ptr);
#endif
}

LweBootstrapKey::LweBootstrapKey(
    Message<concreteprotocol::LweBootstrapKeyInfo> info, LweSecretKey &inputKey,
    LweSecretKey &outputKey, EncryptionCSPRNG &csprng)
    : LweBootstrapKey(info) {
  assert(inputKey.getInfo().asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getInputLweDimension());
  assert(outputKey.getInfo().asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getGlweDimension() *
             info.asReader().getParams().getPolynomialSize());

  auto params = info.asReader().getParams();
  auto compression = info.asReader().getCompression();

  switch (compression) {
  case concreteprotocol::Compression::NONE:
    buffer->resize(concrete_cpu_bootstrap_key_size_u64(
        params.getLevelCount(), params.getGlweDimension(),
        params.getPolynomialSize(), params.getInputLweDimension()));
    concrete_cpu_init_lwe_bootstrap_key_u64(
        buffer->data(), inputKey.getBuffer().data(),
        outputKey.getBuffer().data(), params.getInputLweDimension(),
        params.getPolynomialSize(), params.getGlweDimension(),
        params.getLevelCount(), params.getBaseLog(), params.getVariance(),
        Parallelism::Rayon, csprng.ptr);
    break;
  case concreteprotocol::Compression::SEED:
    compressedBuffer->resize(
        concrete_cpu_seeded_bootstrap_key_size_u64(
            params.getLevelCount(), params.getGlweDimension(),
            params.getPolynomialSize(), params.getInputLweDimension()) +
        2 /* For the seed*/);
    struct Uint128 seed;
    csprng::getRandomSeed(&seed);
    writeSeed(seed, *compressedBuffer);
    concrete_cpu_init_seeded_lwe_bootstrap_key_u64(
        compressedBuffer->data() + 2, inputKey.getBuffer().data(),
        outputKey.getBuffer().data(), params.getInputLweDimension(),
        params.getPolynomialSize(), params.getGlweDimension(),
        params.getLevelCount(), params.getBaseLog(), seed, params.getVariance(),
        Parallelism::Rayon);
    break;
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
};

void LweBootstrapKey::_decompress() {
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    return;
  case concreteprotocol::Compression::SEED: {
    auto params = info.asReader().getParams();
    buffer->resize(concrete_cpu_bootstrap_key_size_u64(
        params.getLevelCount(), params.getGlweDimension(),
        params.getPolynomialSize(), params.getInputLweDimension()));
    struct Uint128 seed;
    readSeed(seed, *compressedBuffer);
    concrete_cpu_decompress_seeded_lwe_bootstrap_key_u64(
        buffer->data(), compressedBuffer->data() + 2,
        params.getInputLweDimension(), params.getPolynomialSize(),
        params.getGlweDimension(), params.getLevelCount(), params.getBaseLog(),
        seed, Parallelism::Rayon);
    return;
  }
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
}

LweKeyswitchKey::LweKeyswitchKey(
    Message<concreteprotocol::LweKeyswitchKeyInfo> info, LweSecretKey &inputKey,
    LweSecretKey &outputKey, EncryptionCSPRNG &csprng)
    : LweKeyswitchKey(info) {
  assert(inputKey.getInfo().asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getInputLweDimension());
  assert(outputKey.getInfo().asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getOutputLweDimension());

  auto params = info.asReader().getParams();
  auto compression = info.asReader().getCompression();

  switch (compression) {
  case concreteprotocol::Compression::NONE:
    buffer->resize(concrete_cpu_keyswitch_key_size_u64(
        params.getLevelCount(), params.getInputLweDimension(),
        params.getOutputLweDimension()));
    concrete_cpu_init_lwe_keyswitch_key_u64(
        buffer->data(), inputKey.getBuffer().data(),
        outputKey.getBuffer().data(), params.getInputLweDimension(),
        params.getOutputLweDimension(), params.getLevelCount(),
        params.getBaseLog(), params.getVariance(), csprng.ptr);
    return;
  case concreteprotocol::Compression::SEED:
    compressedBuffer->resize(
        concrete_cpu_seeded_keyswitch_key_size_u64(
            params.getLevelCount(), params.getInputLweDimension()) +
        2 /* for seed*/);
    struct Uint128 seed;
    csprng::getRandomSeed(&seed);
    writeSeed(seed, *compressedBuffer);
    concrete_cpu_init_seeded_lwe_keyswitch_key_u64(
        compressedBuffer->data() + 2, inputKey.getBuffer().data(),
        outputKey.getBuffer().data(), params.getInputLweDimension(),
        params.getOutputLweDimension(), params.getLevelCount(),
        params.getBaseLog(), seed, params.getVariance());
    return;
  default:
    assert(false && "Unsupported compression type for keyswitch key");
    break;
  }
}

void LweKeyswitchKey::_decompress() {
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    return;
  case concreteprotocol::Compression::SEED: {
    auto params = info.asReader().getParams();
    buffer->resize(concrete_cpu_keyswitch_key_size_u64(
        params.getLevelCount(), params.getInputLweDimension(),
        params.getOutputLweDimension()));
    struct Uint128 seed;
    readSeed(seed, *compressedBuffer);
    concrete_cpu_decompress_seeded_lwe_keyswitch_key_u64(
        buffer->data(), compressedBuffer->data() + 2,
        params.getInputLweDimension(), params.getOutputLweDimension(),
        params.getLevelCount(), params.getBaseLog(), seed, Parallelism::Rayon);
    return;
  }
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
}

PackingKeyswitchKey::PackingKeyswitchKey(
    Message<concreteprotocol::PackingKeyswitchKeyInfo> info,
    LweSecretKey &inputKey, LweSecretKey &outputKey, EncryptionCSPRNG &csprng)
    : PackingKeyswitchKey(info) {
  assert(info.asReader().getParams().getGlweDimension() *
             info.asReader().getParams().getPolynomialSize() ==
         outputKey.getInfo().asReader().getParams().getLweDimension());

  // Allocate the buffer
  auto params = info.asReader().getParams();
  auto bufferSize = concrete_cpu_lwe_packing_keyswitch_key_size(
                        params.getGlweDimension(), params.getPolynomialSize(),
                        params.getLevelCount(), params.getInputLweDimension()) *
                    (params.getGlweDimension() + 1);
  buffer = std::make_shared<std::vector<uint64_t>>();
  (*buffer).resize(bufferSize);

  // We copy the information.
  this->info = info;

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
      buffer->data(), inputKey.getBuffer().data(), outputKey.getBuffer().data(),
      params.getInputLweDimension(), params.getPolynomialSize(),
      params.getGlweDimension(), params.getLevelCount(), params.getBaseLog(),
      params.getVariance(), Parallelism::Rayon, csprng.ptr);
}

} // namespace keys
} // namespace concretelang

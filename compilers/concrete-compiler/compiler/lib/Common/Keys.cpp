
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
                           SecretCSPRNG &csprng) {
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

LweSecretKey
LweSecretKey::fromProto(const Message<concreteprotocol::LweSecretKey> &proto) {
  return fromProto(proto.asReader());
}

LweSecretKey
LweSecretKey::fromProto(concreteprotocol::LweSecretKey::Reader reader) {

  auto info = Message<concreteprotocol::LweSecretKeyInfo>(reader.getInfo());
  auto vector = protoPayloadToSharedVector<uint64_t>(reader.getPayload());
  return LweSecretKey(vector, info);
}

Message<concreteprotocol::LweSecretKey> LweSecretKey::toProto() const {
  return keyToProto<concreteprotocol::LweSecretKey,
                    concreteprotocol::LweSecretKeyInfo, LweSecretKey>(*this);
}

const uint64_t *LweSecretKey::getRawPtr() const { return this->buffer->data(); }

size_t LweSecretKey::getSize() const { return this->buffer->size(); }

const Message<concreteprotocol::LweSecretKeyInfo> &
LweSecretKey::getInfo() const {
  return this->info;
}

const std::vector<uint64_t> &LweSecretKey::getBuffer() const {
  return *this->buffer;
}

LweBootstrapKey::LweBootstrapKey(
    Message<concreteprotocol::LweBootstrapKeyInfo> info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    EncryptionCSPRNG &csprng)
    : LweBootstrapKey(info) {
  assert(inputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getInputLweDimension());
  assert(outputKey.info.asReader().getParams().getLweDimension() ==
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
        buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
        params.getInputLweDimension(), params.getPolynomialSize(),
        params.getGlweDimension(), params.getLevelCount(), params.getBaseLog(),
        params.getVariance(), Parallelism::Rayon, csprng.ptr);
    break;
  case concreteprotocol::Compression::SEED:
    seededBuffer->resize(concrete_cpu_seeded_bootstrap_key_size_u64(
                             params.getLevelCount(), params.getGlweDimension(),
                             params.getPolynomialSize(),
                             params.getInputLweDimension()) +
                         2 /* For the seed*/);
    struct Uint128 seed;
    csprng::getRandomSeed(&seed);
    writeSeed(seed, *seededBuffer);
    concrete_cpu_init_seeded_lwe_bootstrap_key_u64(
        seededBuffer->data() + 2, inputKey.buffer->data(),
        outputKey.buffer->data(), params.getInputLweDimension(),
        params.getPolynomialSize(), params.getGlweDimension(),
        params.getLevelCount(), params.getBaseLog(), seed, params.getVariance(),
        Parallelism::Rayon);
    break;
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
};

LweBootstrapKey LweBootstrapKey::fromProto(
    const Message<concreteprotocol::LweBootstrapKey> &key) {
  return fromProto(key.asReader());
}

LweBootstrapKey
LweBootstrapKey::fromProto(concreteprotocol::LweBootstrapKey::Reader reader) {
  auto info = Message<concreteprotocol::LweBootstrapKeyInfo>(reader.getInfo());
  auto vector = protoPayloadToSharedVector<uint64_t>(reader.getPayload());
  LweBootstrapKey key(info);
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    key.buffer = vector;
    break;
  case concreteprotocol::Compression::SEED:
    key.seededBuffer = vector;
    break;
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
  return key;
}

Message<concreteprotocol::LweBootstrapKey> LweBootstrapKey::toProto() const {
  return keyToProto<concreteprotocol::LweBootstrapKey,
                    concreteprotocol::LweBootstrapKeyInfo, LweBootstrapKey>(
      *this);
}

const std::vector<uint64_t> &LweBootstrapKey::getBuffer() {
  decompress();
  return *buffer;
}

const std::vector<uint64_t> &LweBootstrapKey::getTransportBuffer() const {
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    return *buffer;
  case concreteprotocol::Compression::SEED:
    assert(!seededBuffer->empty());
    return *seededBuffer;
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
}

const Message<concreteprotocol::LweBootstrapKeyInfo> &
LweBootstrapKey::getInfo() const {
  return this->info;
}

void LweBootstrapKey::decompress() {
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    return;
  case concreteprotocol::Compression::SEED: {
    if (*decompressed)
      return;
    const std::lock_guard<std::mutex> guard(*decompress_mutext);
    if (*decompressed)
      return;
    auto params = info.asReader().getParams();
    buffer->resize(concrete_cpu_bootstrap_key_size_u64(
        params.getLevelCount(), params.getGlweDimension(),
        params.getPolynomialSize(), params.getInputLweDimension()));
    struct Uint128 seed;
    readSeed(seed, *seededBuffer);
    concrete_cpu_decompress_seeded_lwe_bootstrap_key_u64(
        buffer->data(), seededBuffer->data() + 2, params.getInputLweDimension(),
        params.getPolynomialSize(), params.getGlweDimension(),
        params.getLevelCount(), params.getBaseLog(), seed, Parallelism::Rayon);
    *decompressed = true;
    return;
  }
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
}

LweKeyswitchKey::LweKeyswitchKey(
    Message<concreteprotocol::LweKeyswitchKeyInfo> info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    EncryptionCSPRNG &csprng)
    : LweKeyswitchKey(info) {
  assert(inputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getInputLweDimension());
  assert(outputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getOutputLweDimension());

  auto params = info.asReader().getParams();
  auto compression = info.asReader().getCompression();

  switch (compression) {
  case concreteprotocol::Compression::NONE:
    buffer->resize(concrete_cpu_keyswitch_key_size_u64(
        params.getLevelCount(), params.getInputLweDimension(),
        params.getOutputLweDimension()));
    concrete_cpu_init_lwe_keyswitch_key_u64(
        buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
        params.getInputLweDimension(), params.getOutputLweDimension(),
        params.getLevelCount(), params.getBaseLog(), params.getVariance(),
        csprng.ptr);
    return;
  case concreteprotocol::Compression::SEED:
    seededBuffer->resize(
        concrete_cpu_seeded_keyswitch_key_size_u64(
            params.getLevelCount(), params.getInputLweDimension()) +
        2 /* for seed*/);
    struct Uint128 seed;
    csprng::getRandomSeed(&seed);
    writeSeed(seed, *seededBuffer);
    concrete_cpu_init_seeded_lwe_keyswitch_key_u64(
        seededBuffer->data() + 2, inputKey.buffer->data(),
        outputKey.buffer->data(), params.getInputLweDimension(),
        params.getOutputLweDimension(), params.getLevelCount(),
        params.getBaseLog(), seed, params.getVariance());
    return;
  default:
    assert(false && "Unsupported compression type for keyswitch key");
    break;
  }
}

LweKeyswitchKey LweKeyswitchKey::fromProto(
    const Message<concreteprotocol::LweKeyswitchKey> &proto) {
  return fromProto(proto.asReader());
}

LweKeyswitchKey
LweKeyswitchKey::fromProto(concreteprotocol::LweKeyswitchKey::Reader reader) {
  auto info = Message<concreteprotocol::LweKeyswitchKeyInfo>(reader.getInfo());
  auto vector = protoPayloadToSharedVector<uint64_t>(reader.getPayload());
  LweKeyswitchKey key(info);
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    key.buffer = vector;
    break;
  case concreteprotocol::Compression::SEED:
    key.seededBuffer = vector;
    break;
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
  return key;
}

Message<concreteprotocol::LweKeyswitchKey> LweKeyswitchKey::toProto() const {
  return keyToProto<concreteprotocol::LweKeyswitchKey,
                    concreteprotocol::LweKeyswitchKeyInfo, LweKeyswitchKey>(
      *this);
}

const Message<concreteprotocol::LweKeyswitchKeyInfo> &
LweKeyswitchKey::getInfo() const {
  return this->info;
}

const std::vector<uint64_t> &LweKeyswitchKey::getBuffer() {
  decompress();
  return *buffer;
}

const std::vector<uint64_t> &LweKeyswitchKey::getTransportBuffer() const {
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    return *buffer;
  case concreteprotocol::Compression::SEED:
    assert(!seededBuffer->empty());
    return *seededBuffer;
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
}

void LweKeyswitchKey::decompress() {
  switch (info.asReader().getCompression()) {
  case concreteprotocol::Compression::NONE:
    return;
  case concreteprotocol::Compression::SEED: {
    if (*decompressed)
      return;
    const std::lock_guard<std::mutex> guard(*decompress_mutext);
    if (*decompressed)
      return;
    auto params = info.asReader().getParams();
    buffer->resize(concrete_cpu_keyswitch_key_size_u64(
        params.getLevelCount(), params.getInputLweDimension(),
        params.getOutputLweDimension()));
    struct Uint128 seed;
    readSeed(seed, *seededBuffer);
    concrete_cpu_decompress_seeded_lwe_keyswitch_key_u64(
        buffer->data(), seededBuffer->data() + 2, params.getInputLweDimension(),
        params.getOutputLweDimension(), params.getLevelCount(),
        params.getBaseLog(), seed, Parallelism::Rayon);
    *decompressed = true;
    return;
  }
  default:
    assert(false && "Unsupported compression type for bootstrap key");
  }
}

PackingKeyswitchKey::PackingKeyswitchKey(
    Message<concreteprotocol::PackingKeyswitchKeyInfo> info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    EncryptionCSPRNG &csprng) {
  assert(info.asReader().getParams().getGlweDimension() *
             info.asReader().getParams().getPolynomialSize() ==
         outputKey.info.asReader().getParams().getLweDimension());

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
      buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
      params.getInputLweDimension(), params.getPolynomialSize(),
      params.getGlweDimension(), params.getLevelCount(), params.getBaseLog(),
      params.getVariance(), Parallelism::Rayon, csprng.ptr);
}

PackingKeyswitchKey PackingKeyswitchKey::fromProto(
    const Message<concreteprotocol::PackingKeyswitchKey> &proto) {
  return fromProto(proto.asReader());
}

PackingKeyswitchKey PackingKeyswitchKey::fromProto(
    concreteprotocol::PackingKeyswitchKey::Reader reader) {
  auto info =
      Message<concreteprotocol::PackingKeyswitchKeyInfo>(reader.getInfo());
  auto vector = protoPayloadToSharedVector<uint64_t>(reader.getPayload());
  return PackingKeyswitchKey(vector, info);
}

Message<concreteprotocol::PackingKeyswitchKey>
PackingKeyswitchKey::toProto() const {
  return keyToProto<concreteprotocol::PackingKeyswitchKey,
                    concreteprotocol::PackingKeyswitchKeyInfo,
                    PackingKeyswitchKey>(*this);
}

const uint64_t *PackingKeyswitchKey::getRawPtr() const {
  return this->buffer->data();
}

size_t PackingKeyswitchKey::getSize() const { return this->buffer->size(); }

const Message<concreteprotocol::PackingKeyswitchKeyInfo> &
PackingKeyswitchKey::getInfo() const {
  return this->info;
}

const std::vector<uint64_t> &PackingKeyswitchKey::getBuffer() const {
  return *this->buffer;
}

} // namespace keys
} // namespace concretelang

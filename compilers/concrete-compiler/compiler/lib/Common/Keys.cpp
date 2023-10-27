
// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
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

using concretelang::csprng::CSPRNG;
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
  proto.setPayload(vectorToProtoPayload(key.getBuffer()).asReader());
  return std::move(output);
}

LweSecretKey::LweSecretKey(Message<concreteprotocol::LweSecretKeyInfo> info,
                           CSPRNG &csprng) {
  // Allocate the buffer
  buffer = std::make_shared<std::vector<uint64_t>>(
      info.asReader().getParams().getLweDimension());

  // We copy the informations.
  this->info = info;

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  // In insecure debug mode, the secret key is filled with zeros.
  getApproval();
  std::fill(buffer->begin(), buffer->end(), 0);
#else
  // Initialize the lwe secret key buffer
  concrete_cpu_init_secret_key_u64(
      buffer->data(), info.asReader().getParams().getLweDimension(), csprng.ptr,
      csprng.vtable);
#endif
}

LweSecretKey
LweSecretKey::fromProto(const Message<concreteprotocol::LweSecretKey> &proto) {

  auto info =
      Message<concreteprotocol::LweSecretKeyInfo>(proto.asReader().getInfo());
  auto vector =
      protoPayloadToSharedVector<uint64_t>(proto.asReader().getPayload());
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
    CSPRNG &csprng) {
  assert(info.asReader().getCompression() ==
         concreteprotocol::Compression::NONE);
  assert(inputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getInputLweDimension());
  assert(outputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getGlweDimension() *
             info.asReader().getParams().getPolynomialSize());

  // Allocate the buffer
  auto params = info.asReader().getParams();
  auto bufferSize = concrete_cpu_bootstrap_key_size_u64(
      params.getLevelCount(), params.getGlweDimension(),
      params.getPolynomialSize(), params.getInputLweDimension());
  buffer = std::make_shared<std::vector<uint64_t>>();
  (*buffer).resize(bufferSize);

  // We copy the informations.
  this->info = info;

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_bootstrap_key_u64(
      buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
      params.getInputLweDimension(), params.getPolynomialSize(),
      params.getGlweDimension(), params.getLevelCount(), params.getBaseLog(),
      params.getVariance(), Parallelism::Rayon, csprng.ptr, csprng.vtable);
};

LweBootstrapKey LweBootstrapKey::fromProto(
    const Message<concreteprotocol::LweBootstrapKey> &proto) {
  assert(proto.asReader().getInfo().getCompression() ==
         concreteprotocol::Compression::NONE);
  auto info = Message<concreteprotocol::LweBootstrapKeyInfo>(
      proto.asReader().getInfo());
  auto vector =
      protoPayloadToSharedVector<uint64_t>(proto.asReader().getPayload());
  return LweBootstrapKey(vector, info);
}

Message<concreteprotocol::LweBootstrapKey> LweBootstrapKey::toProto() const {
  return keyToProto<concreteprotocol::LweBootstrapKey,
                    concreteprotocol::LweBootstrapKeyInfo, LweBootstrapKey>(
      *this);
}

const uint64_t *LweBootstrapKey::getRawPtr() const {
  return this->buffer->data();
}

size_t LweBootstrapKey::getSize() const { return this->buffer->size(); }

const Message<concreteprotocol::LweBootstrapKeyInfo> &
LweBootstrapKey::getInfo() const {
  return this->info;
}

const std::vector<uint64_t> &LweBootstrapKey::getBuffer() const {
  return *this->buffer;
}

LweKeyswitchKey::LweKeyswitchKey(
    Message<concreteprotocol::LweKeyswitchKeyInfo> info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    CSPRNG &csprng) {
  assert(info.asReader().getCompression() ==
         concreteprotocol::Compression::NONE);
  assert(inputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getInputLweDimension());
  assert(outputKey.info.asReader().getParams().getLweDimension() ==
         info.asReader().getParams().getOutputLweDimension());

  // Allocate the buffer
  auto params = info.asReader().getParams();
  auto bufferSize = concrete_cpu_keyswitch_key_size_u64(
      params.getLevelCount(), params.getBaseLog(),
      params.getInputLweDimension(), params.getOutputLweDimension());
  buffer = std::make_shared<std::vector<uint64_t>>();
  (*buffer).resize(bufferSize);

  // We copy the informations.
  this->info = info;

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_keyswitch_key_u64(
      buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
      params.getInputLweDimension(), params.getOutputLweDimension(),
      params.getLevelCount(), params.getBaseLog(), params.getVariance(),
      csprng.ptr, csprng.vtable);
}

LweKeyswitchKey LweKeyswitchKey::fromProto(
    const Message<concreteprotocol::LweKeyswitchKey> &proto) {
  assert(proto.asReader().getInfo().getCompression() ==
         concreteprotocol::Compression::NONE);
  auto info = Message<concreteprotocol::LweKeyswitchKeyInfo>(
      proto.asReader().getInfo());
  auto vector =
      protoPayloadToSharedVector<uint64_t>(proto.asReader().getPayload());
  return LweKeyswitchKey(vector, info);
}

Message<concreteprotocol::LweKeyswitchKey> LweKeyswitchKey::toProto() const {
  return keyToProto<concreteprotocol::LweKeyswitchKey,
                    concreteprotocol::LweKeyswitchKeyInfo, LweKeyswitchKey>(
      *this);
}

const uint64_t *LweKeyswitchKey::getRawPtr() const {
  return this->buffer->data();
}

size_t LweKeyswitchKey::getSize() const { return this->buffer->size(); }

const Message<concreteprotocol::LweKeyswitchKeyInfo> &
LweKeyswitchKey::getInfo() const {
  return this->info;
}

const std::vector<uint64_t> &LweKeyswitchKey::getBuffer() const {
  return *this->buffer;
}

PackingKeyswitchKey::PackingKeyswitchKey(
    Message<concreteprotocol::PackingKeyswitchKeyInfo> info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    CSPRNG &csprng) {
  assert(info.asReader().getCompression() ==
         concreteprotocol::Compression::NONE);
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

  // We copy the informations.
  this->info = info;

  // Initialize the keyswitch key buffer
  concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
      buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
      params.getInputLweDimension(), params.getPolynomialSize(),
      params.getGlweDimension(), params.getLevelCount(), params.getBaseLog(),
      params.getVariance(), Parallelism::Rayon, csprng.ptr, csprng.vtable);
}

PackingKeyswitchKey PackingKeyswitchKey::fromProto(
    const Message<concreteprotocol::PackingKeyswitchKey> &proto) {
  assert(proto.asReader().getInfo().getCompression() ==
         concreteprotocol::Compression::NONE);
  auto info = Message<concreteprotocol::PackingKeyswitchKeyInfo>(
      proto.asReader().getInfo());
  auto vector =
      protoPayloadToSharedVector<uint64_t>(proto.asReader().getPayload());
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


// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concrete-cpu.h"
#include "concrete-protocol.pb.h"
#include "concretelang/Common/Keys.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include <cstddef>
#include <memory>
#include <stdlib.h>

using concretelang::csprng::CSPRNG;
using concretelang::protocol::protoDataToVector;
using concretelang::protocol::vectorToProtoData;

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
inline void getApproval() {
  std::cerr << "DANGER: You are using an empty unsecure secret keys. Enter "
               "\"y\" to continue: ";
  char answer;
  std::cin >> answer;
  if (answer != 'y') {
    std::abort();
  }
}
#endif

namespace concretelang {
namespace keys {

LweSecretKey::LweSecretKey(const concreteprotocol::LweSecretKeyInfo &info,
                           SecretCSPRNG &csprng) {
  // Allocate the buffer
  buffer = std::make_shared<std::vector<uint64_t>>(info.params().lwedimension());
  
  // We copy the informations.
  this->info = info;

#ifdef CONCRETELANG_GENERATE_UNSECURE_SECRET_KEYS
  // In insecure debug mode, the secret key is filled with zeros.
  getApproval();
  for (uint64_t &val : *_buffer) {
    val = 0;
  }
#else
  // Initialize the lwe secret key buffer
  concrete_cpu_init_secret_key_u64(buffer->data(), info.params().lwedimension(),
                                   csprng.ptr);
#endif
}

LweSecretKey
LweSecretKey::fromProto(const concreteprotocol::LweSecretKey &proto) {
  LweSecretKey output;
  output.buffer = std::make_shared<std::vector<uint64_t>>(protoDataToVector<uint64_t>(proto.data()));
  output.info = proto.info();
  auto expectedBufferSize = proto.info().params().lwedimension();
  assert(output.buffer->size() == expectedBufferSize);
  return output;
}

concreteprotocol::LweSecretKey LweSecretKey::toProto() {
  concreteprotocol::LweSecretKey output;
  output.set_allocated_info(new concreteprotocol::LweSecretKeyInfo(this->info));
  output.set_allocated_data(vectorToProtoData(*this->buffer));
  return output;
}

const uint64_t *LweSecretKey::getRawPtr() const { return this->buffer->data(); }

size_t LweSecretKey::getSize() { return this->buffer->size(); }

const concreteprotocol::LweSecretKeyInfo &LweSecretKey::getInfo() {
  return this->info;
}

LweBootstrapKey::LweBootstrapKey(
    const concreteprotocol::LweBootstrapKeyInfo &info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    EncryptionCSPRNG &csprng)
  : info(info) {
  assert(info.compression() == concreteprotocol::Compression::none
	 || info.compression() == concreteprotocol::Compression::seed);
  assert(inputKey.info.params().lwedimension() ==
         info.params().inputlwedimension());
  assert(outputKey.info.params().lwedimension() ==
         info.params().glwedimension() * info.params().polynomialsize());

  // Allocate the buffer
  auto params = info.params();
  size_t bufferSize;
  if (info.compression() == concreteprotocol::Compression::none) {
    bufferSize = concrete_cpu_bootstrap_key_size_u64(params.levelcount(),
					       params.glwedimension(),
					       params.polynomialsize(),
					       params.inputlwedimension());
  } else if (info.compression() == concreteprotocol::Compression::seed) {
    bufferSize = concrete_cpu_seeded_bootstrap_key_size_u64(params.levelcount(),
						      params.glwedimension(),
						      params.polynomialsize(),
						     params.inputlwedimension());
  }

  buffer = std::make_shared<std::vector<uint64_t>>(bufferSize);

  // Initialize the bootstrap key buffer

  if (info.compression() == concreteprotocol::Compression::none) {
    concrete_cpu_init_lwe_bootstrap_key_u64(
        buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
        params.inputlwedimension(), params.polynomialsize(),
	params.glwedimension(), params.levelcount(), params.baselog(),
	params.variance(), Parallelism::Rayon, csprng.ptr);
  } else if (info.compression() == concreteprotocol::Compression::seed) {
    struct Uint128 u128;
    csprng::getRandomSeed(&u128);
    concrete_cpu_init_seeded_lwe_bootstrap_key_u64(
        buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
        params.inputlwedimension(), params.polynomialsize(),
	params.glwedimension(), params.levelcount(), params.baselog(),
	u128, params.variance(), Parallelism::Rayon);
  }
};

LweBootstrapKey
LweBootstrapKey::fromProto(const concreteprotocol::LweBootstrapKey &proto) {
  assert(proto.info().compression() == concreteprotocol::Compression::none
	 || proto.info().compression() == concreteprotocol::Compression::seed);
  LweBootstrapKey output;
  output.buffer = std::make_shared<std::vector<uint64_t>>(protoDataToVector<uint64_t>(proto.data()));
  output.info = proto.info();
  auto params = proto.info().params();
  size_t expectedBufferSize;
  if (proto.info().compression() == concreteprotocol::Compression::none) {
    expectedBufferSize = concrete_cpu_bootstrap_key_size_u64(
      params.levelcount(), params.glwedimension(), params.polynomialsize(),
      params.inputlwedimension());
  } else if (proto.info().compression() == concreteprotocol::Compression::seed) {
      expectedBufferSize = concrete_cpu_seeded_bootstrap_key_size_u64(params.levelcount(),
						      params.glwedimension(),
						      params.polynomialsize(),
						     params.inputlwedimension());
  } else expectedBufferSize = 0;
  assert(output.buffer->size() == expectedBufferSize);
  return output;
}

concreteprotocol::LweBootstrapKey LweBootstrapKey::toProto() {
  concreteprotocol::LweBootstrapKey output;
  output.set_allocated_info(
      new concreteprotocol::LweBootstrapKeyInfo(this->info));
  output.set_allocated_data(vectorToProtoData(*this->buffer));
  return output;
}

const uint64_t *LweBootstrapKey::getRawPtr() const {
  return this->buffer->data();
}

size_t LweBootstrapKey::getSize() { return this->buffer->size(); }

const concreteprotocol::LweBootstrapKeyInfo &LweBootstrapKey::getInfo() {
  return this->info;
}

LweKeyswitchKey::LweKeyswitchKey(
    const concreteprotocol::LweKeyswitchKeyInfo &info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    EncryptionCSPRNG &csprng)
  : info(info) {
  assert(info.compression() == concreteprotocol::Compression::none
	 || info.compression() == concreteprotocol::Compression::seed);
  assert(inputKey.info.params().lwedimension() ==
         info.params().inputlwedimension());
  assert(outputKey.info.params().lwedimension() ==
         info.params().outputlwedimension());

  // Allocate the buffer
  auto params = info.params();
  size_t bufferSize;
  if (info.compression() == concreteprotocol::Compression::none) {
    bufferSize = concrete_cpu_keyswitch_key_size_u64(
						     params.levelcount(), params.inputlwedimension(), params.outputlwedimension());
  } else if (info.compression() == concreteprotocol::Compression::seed) {
    bufferSize = concrete_cpu_seeded_keyswitch_key_size_u64(params.levelcount(),
							    params.inputlwedimension());
  }

  buffer = std::make_shared<std::vector<uint64_t>>(bufferSize);

  // Initialize the keyswitch key buffer
  if (info.compression() == concreteprotocol::Compression::none) {
    concrete_cpu_init_lwe_keyswitch_key_u64(
        buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
        params.inputlwedimension(), params.outputlwedimension(),
	params.levelcount(), params.baselog(), params.variance(), csprng.ptr);
  } else if (info.compression() == concreteprotocol::Compression::seed) {
    struct Uint128 u128;
    csprng::getRandomSeed(&u128);
    concrete_cpu_init_seeded_lwe_keyswitch_key_u64(
        buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
	params.inputlwedimension(), params.outputlwedimension(),
	params.levelcount(), params.baselog(),
 u128,params.variance());
  }
}

LweKeyswitchKey
LweKeyswitchKey::fromProto(const concreteprotocol::LweKeyswitchKey &proto) {
  assert(proto.info().compression() == concreteprotocol::Compression::none
	 || proto.info().compression() == concreteprotocol::Compression::seed);
  LweKeyswitchKey output;
  output.buffer = std::make_shared<std::vector<uint64_t>>(protoDataToVector<uint64_t>(proto.data()));
  output.info = proto.info();
  auto params = proto.info().params();
  size_t expectedBufferSize;
  if (proto.info().compression() == concreteprotocol::Compression::none) {
    expectedBufferSize = concrete_cpu_keyswitch_key_size_u64(params.levelcount(), params.inputlwedimension(), params.outputlwedimension());
  } else if (proto.info().compression() == concreteprotocol::Compression::seed) {
    expectedBufferSize = concrete_cpu_seeded_keyswitch_key_size_u64(params.levelcount(),
							    params.inputlwedimension());
  } else expectedBufferSize = 0;
  assert(output.buffer->size() == expectedBufferSize);
  return output;
}

concreteprotocol::LweKeyswitchKey LweKeyswitchKey::toProto() {
  concreteprotocol::LweKeyswitchKey output;
  output.set_allocated_info(
      new concreteprotocol::LweKeyswitchKeyInfo(this->info));
  output.set_allocated_data(vectorToProtoData(*this->buffer));
  return output;
}

const uint64_t *LweKeyswitchKey::getRawPtr() const {
  return this->buffer->data();
}
size_t LweKeyswitchKey::getSize() { return this->buffer->size(); }
const concreteprotocol::LweKeyswitchKeyInfo &LweKeyswitchKey::getInfo() {
  return this->info;
}

PackingKeyswitchKey::PackingKeyswitchKey(
    const concreteprotocol::PackingKeyswitchKeyInfo &info,
    const LweSecretKey &inputKey, const LweSecretKey &outputKey,
    EncryptionCSPRNG &csprng) {
  assert(info.compression() == concreteprotocol::Compression::none);
  assert(info.params().lwedimension() == inputKey.info.params().lwedimension());
  assert(info.params().glwedimension() * info.params().polynomialsize() ==
         outputKey.info.params().lwedimension());

  // Allocate the buffer
  auto params = info.params();
  auto bufferSize = concrete_cpu_lwe_packing_keyswitch_key_size(
                        params.glwedimension(), params.polynomialsize(),
                        params.levelcount(), params.lwedimension()) *
                    (params.glwedimension() + 1);
  buffer = std::make_shared<std::vector<uint64_t>>(bufferSize);

  // We copy the informations.
  this->info = info;

  // Initialize the packing keyswitch key buffer
  concrete_cpu_init_lwe_circuit_bootstrap_private_functional_packing_keyswitch_keys_u64(
      buffer->data(), inputKey.buffer->data(), outputKey.buffer->data(),
      params.lwedimension(), params.polynomialsize(), params.glwedimension(),
      params.levelcount(), params.baselog(), params.variance(),
      Parallelism::Rayon, csprng.ptr);
}

PackingKeyswitchKey PackingKeyswitchKey::fromProto(
    const concreteprotocol::PackingKeyswitchKey &proto) {
  assert(proto.info().compression() == concreteprotocol::Compression::none);
  PackingKeyswitchKey output;
  output.buffer = std::make_shared<std::vector<uint64_t>>(protoDataToVector<uint64_t>(proto.data()));
  output.info = proto.info();
  auto params = proto.info().params();
  auto expectedBufferSize = concrete_cpu_lwe_packing_keyswitch_key_size(
                                params.glwedimension(), params.polynomialsize(),
                                params.levelcount(), params.lwedimension()) *
                            (params.glwedimension() + 1);
  assert(output.buffer->size() == expectedBufferSize);
  return output;
}

concreteprotocol::PackingKeyswitchKey PackingKeyswitchKey::toProto() {
  concreteprotocol::PackingKeyswitchKey output;
  output.set_allocated_info(
      new concreteprotocol::PackingKeyswitchKeyInfo(this->info));
  output.set_allocated_data(vectorToProtoData(*this->buffer));
  return output;
}

const uint64_t *PackingKeyswitchKey::getRawPtr() const {
  return this->buffer->data();
}

size_t PackingKeyswitchKey::getSize() { return this->buffer->size(); }

const concreteprotocol::PackingKeyswitchKeyInfo &
PackingKeyswitchKey::getInfo() {
  return this->info;
}

} // namespace keys
} // namespace concretelang

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_TRANSFORMERS_H
#define CONCRETELANG_COMMON_TRANSFORMERS_H

#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include <memory>
#include <stdlib.h>

using concretelang::error::Result;
using concretelang::keysets::ClientKeyset;
using concretelang::values::Tensor;
using concretelang::values::TransportValue;
using concretelang::values::Value;

/// \brief simulate the encryption of a value by adding noise
///
/// \param message encoded message to encrypt
/// \param lwe_dim
/// \param csprng used to generate noise during encryption
/// \return noisy plaintext
uint64_t sim_encrypt_lwe_u64(uint64_t message, uint32_t lwe_dim,
                             Csprng *encrypt_csprng);

namespace concretelang {
namespace transformers {

/// A type for input transformers, that is, functions running on the client
/// side, that prepare a Value to be sent to the server as a TransportValue.
typedef std::function<Result<TransportValue>(Value)> InputTransformer;

/// A type for output transformers, that is, functions running on the client
/// side, that process a TransportValue fetched from the server to be used as a
/// Value.
typedef std::function<Result<Value>(TransportValue)> OutputTransformer;

/// A type for arguments transformers, that is, functions running on the server
/// side, that transform a TransportValue fetched from the client, to be used as
/// argument in a circuit call.
typedef std::function<Result<Value>(TransportValue)> ArgTransformer;

/// A type for return transformers, that is, functions running on the server
/// side, that transform a value returned from circuit call into a
/// TransportValue to be sent to the client.
typedef std::function<Result<TransportValue>(Value)> ReturnTransformer;

/// A factory static class that generates transformers.
class TransformerFactory {
public:
  static Result<InputTransformer>
  getIndexInputTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<OutputTransformer>
  getIndexOutputTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<ArgTransformer>
  getIndexArgTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<ReturnTransformer>
  getIndexReturnTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<InputTransformer>
  getPlaintextInputTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<OutputTransformer>
  getPlaintextOutputTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<ArgTransformer>
  getPlaintextArgTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<ReturnTransformer>
  getPlaintextReturnTransformer(Message<concreteprotocol::GateInfo> gateInfo);

  static Result<InputTransformer> getLweCiphertextInputTransformer(
      ClientKeyset keyset, Message<concreteprotocol::GateInfo> gateInfo,
      std::shared_ptr<concretelang::csprng::EncryptionCSPRNG> csprng,
      bool useSimulation);

  static Result<OutputTransformer> getLweCiphertextOutputTransformer(
      ClientKeyset keyset, Message<concreteprotocol::GateInfo> gateInfo,
      bool useSimulation);

  static Result<ArgTransformer>
  getLweCiphertextArgTransformer(Message<concreteprotocol::GateInfo> gateInfo,
                                 bool useSimulation);

  static Result<ReturnTransformer> getLweCiphertextReturnTransformer(
      Message<concreteprotocol::GateInfo> gateInfo, bool useSimulation);
};

} // namespace transformers
} // namespace concretelang

#endif

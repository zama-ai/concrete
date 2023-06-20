// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_COMMON_TRANSFORMERS_H
#define CONCRETELANG_COMMON_TRANSFORMERS_H

#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Common/Keysets.h"
#include <stdlib.h>

using concretelang::error::Result;
using concretelang::values::TransportValue;
using concretelang::values::Value;
using concretelang::values::Tensor;
using concretelang::keysets::ClientKeyset;
using google::protobuf::util::MessageDifferencer;

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
  getIndexInputTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<OutputTransformer>
  getIndexOutputTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<ArgTransformer>
  getIndexArgTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<ReturnTransformer>
  getIndexReturnTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<InputTransformer>
  getPlaintextInputTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<OutputTransformer>
  getPlaintextOutputTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<ArgTransformer>
  getPlaintextArgTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<ReturnTransformer>
  getPlaintextReturnTransformer(concreteprotocol::GateInfo gateInfo);

  static Result<InputTransformer>
  getLweCiphertextInputTransformer(ClientKeyset keyset,
                                   concreteprotocol::GateInfo gateInfo,
                                   CSPRNG &csprng, bool useSimulation);

  static Result<OutputTransformer>
  getLweCiphertextOutputTransformer(ClientKeyset keyset,
                                    concreteprotocol::GateInfo gateInfo, bool useSimulation);

  static Result<ArgTransformer>
  getLweCiphertextArgTransformer(concreteprotocol::GateInfo gateInfo, bool useSimulation);

  static Result<ReturnTransformer>
  getLweCiphertextReturnTransformer(concreteprotocol::GateInfo gateInfo, bool useSimulation);

};

} // namespace transformers
} // namespace concretelang

#endif

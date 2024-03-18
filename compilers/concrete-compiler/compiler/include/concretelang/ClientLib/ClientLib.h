// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CLIENTLIB_REFACTORED_H
#define CONCRETELANG_CLIENTLIB_REFACTORED_H

#include <cassert>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <variant>

#include "boost/outcome.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Common/Csprng.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Transformers.h"
#include "concretelang/Common/Values.h"

using concretelang::error::Result;
using concretelang::keysets::ClientKeyset;
using concretelang::transformers::InputTransformer;
using concretelang::transformers::OutputTransformer;
using concretelang::transformers::TransformerFactory;
using concretelang::values::TransportValue;
using concretelang::values::Value;

namespace concretelang {
namespace clientlib {

class ClientCircuit {

public:
  static Result<ClientCircuit>
  create(const Message<concreteprotocol::CircuitInfo> &info,
         const ClientKeyset &keyset,
         std::shared_ptr<csprng::EncryptionCSPRNG> csprng,
         bool useSimulation = false);

  Result<TransportValue> prepareInput(Value arg, size_t pos);

  Result<Value> processOutput(TransportValue result, size_t pos);

  std::string getName();

  const Message<concreteprotocol::CircuitInfo> &getCircuitInfo();

private:
  ClientCircuit() = delete;
  ClientCircuit(const Message<concreteprotocol::CircuitInfo> &circuitInfo,
                std::vector<InputTransformer> inputTransformers,
                std::vector<OutputTransformer> outputTransformers)
      : circuitInfo(circuitInfo), inputTransformers(inputTransformers),
        outputTransformers(outputTransformers){};

private:
  Message<concreteprotocol::CircuitInfo> circuitInfo;
  std::vector<InputTransformer> inputTransformers;
  std::vector<OutputTransformer> outputTransformers;
};

/// Contains all the context to generate inputs for a server call by the
/// server lib.
class ClientProgram {
public:
  /// Generates a fresh client program with fresh keyset on the first use.
  static Result<ClientProgram>
  create(const Message<concreteprotocol::ProgramInfo> &info,
         const ClientKeyset &keyset,
         std::shared_ptr<csprng::EncryptionCSPRNG> csprng,
         bool useSimulation = false);

  /// Returns a reference to the named client circuit if it exists.
  Result<ClientCircuit> getClientCircuit(std::string circuitName);

private:
  ClientProgram() = default;

private:
  std::vector<ClientCircuit> circuits;
};

} // namespace clientlib
} // namespace concretelang

#endif

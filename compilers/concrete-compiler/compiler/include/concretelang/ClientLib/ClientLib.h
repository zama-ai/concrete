// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
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
#include "concrete-cpu.h"
#include "concrete-protocol.pb.h"
#include "concretelang/Common/Error.h"
#include "concretelang/Common/Protocol.h"
#include "concretelang/Common/Keysets.h"
#include "concretelang/Common/Values.h"
#include "concretelang/Common/Transformers.h"
#include <google/protobuf/util/message_differencer.h>

using concretelang::error::Result;
using concretelang::keysets::ClientKeyset;
using concretelang::values::TransportValue;
using concretelang::values::Value;
using concretelang::transformers::InputTransformer;
using concretelang::transformers::OutputTransformer;
using concretelang::transformers::TransformerFactory;

namespace concretelang {
namespace clientlib {

class ClientCircuit {

public:
  static Result<ClientCircuit> create(const concreteprotocol::CircuitInfo &info,
                                      const ClientKeyset &keyset,
                                      CSPRNG &csprng,
                                      bool useSimulation = false);

  Result<TransportValue> prepareInput(Value arg, size_t pos);

  Result<Value> processOutput(TransportValue result, size_t pos);

  std::string getName();

private:
  ClientCircuit() = default;

private:
  concreteprotocol::CircuitInfo circuitInfo;
  std::vector<InputTransformer> inputTransformers;
  std::vector<OutputTransformer> outputTransformers;
};

/// Contains all the context to generate inputs for a server call by the
/// server lib.
class ClientProgram {
public:
  /// Generates a fresh client program with fresh keyset on the first use.
  static Result<ClientProgram> create(const concreteprotocol::ProgramInfo &info, const ClientKeyset &keyset, CSPRNG& csprng, bool useSimulation = false);

  /// Returns a reference to the named client circuit if it exists.
  Result<std::reference_wrapper<ClientCircuit>> getClientCircuit(std::string circuitName);

private:
  ClientProgram() = default;

private:
  std::vector<ClientCircuit> circuits;
};

} // namespace clientlib
} // namespace concretelang

#endif

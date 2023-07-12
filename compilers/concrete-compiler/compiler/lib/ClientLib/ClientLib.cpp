// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
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
#include "concretelang/ClientLib/ClientLib.h"
#include "concretelang/Common/Csprng.h"
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

   Result<ClientCircuit> ClientCircuit::create(const concreteprotocol::CircuitInfo &info,
                                      const ClientKeyset &keyset,
                                      CSPRNG &csprng,
                                      bool useSimulation) {
    ClientCircuit output;

    for (auto gateInfo : info.inputs()) {
      InputTransformer transformer;
      if (gateInfo.has_index()) {
        OUTCOME_TRY(transformer,
                    TransformerFactory::getIndexInputTransformer(gateInfo));
      } else if (gateInfo.has_plaintext()) {
        OUTCOME_TRY(transformer,
                    TransformerFactory::getPlaintextInputTransformer(gateInfo));
      } else if (gateInfo.has_lweciphertext()) {
        OUTCOME_TRY(transformer,
                    TransformerFactory::getLweCiphertextInputTransformer(
                        keyset, gateInfo, csprng, useSimulation));
      } else {
        return StringError("Malformed input gate info.");
      }
      output.inputTransformers.push_back(transformer);
    }

    for (auto gateInfo : info.outputs()) {
      OutputTransformer transformer;
      if (gateInfo.has_index()) {
        OUTCOME_TRY(transformer,
                    TransformerFactory::getIndexOutputTransformer(gateInfo));
      } else if (gateInfo.has_plaintext()) {
        OUTCOME_TRY(
            transformer,
            TransformerFactory::getPlaintextOutputTransformer(gateInfo));
      } else if (gateInfo.has_lweciphertext()) {
        OUTCOME_TRY(transformer,
                    TransformerFactory::getLweCiphertextOutputTransformer(
                        keyset, gateInfo, useSimulation));
      } else {
        return StringError("Malformed output gate info.");
      }
      output.outputTransformers.push_back(transformer);
    }

    output.circuitInfo = info;

    return output;
  }

  Result<TransportValue> ClientCircuit::prepareInput(Value arg, size_t pos) {
    if (pos > inputTransformers.size()) {
      return StringError("Tried to prepare a Value for incorrect position.");
    }
    return inputTransformers[pos](arg);
  }

  Result<Value> ClientCircuit::processOutput(TransportValue result, size_t pos) {
    if (pos > outputTransformers.size()) {
      return StringError("Tried to process a TransportValue for incorrect position.");
    }
    return outputTransformers[pos](result);
  }

  std::string ClientCircuit::getName() {
    return circuitInfo.name();
  }


  Result<ClientProgram> ClientProgram::create(const concreteprotocol::ProgramInfo &info, const ClientKeyset &keyset, CSPRNG& csprng, bool useSimulation){
    ClientProgram output;
    for (auto circuitInfo : info.circuits()) {
      OUTCOME_TRY(ClientCircuit clientCircuit, ClientCircuit::create(circuitInfo, keyset, csprng, useSimulation));
      output.circuits.push_back(clientCircuit);
    }
    return output;
  }

  Result<ClientCircuit> ClientProgram::getClientCircuit(std::string circuitName){
    for (auto circuit: circuits) {
      if(circuit.getName() == circuitName){
        return circuit;
      }
    }
    return StringError("Tried to get unknown client circuit: `" + circuitName + "`");
  }

} // namespace clientlib
} // namespace concretelang

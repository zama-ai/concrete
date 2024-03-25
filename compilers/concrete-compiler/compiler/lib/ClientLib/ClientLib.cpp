// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
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
#include "concrete-protocol.capnp.h"
#include "concretelang/ClientLib/ClientLib.h"
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

Result<ValueExporter>
ValueExporter::create(const Message<concreteprotocol::CircuitInfo> &info,
                      const ClientKeyset &keyset,
                      std::shared_ptr<csprng::EncryptionCSPRNG> csprng,
                      bool useSimulation) {

  auto inputTransformers = std::vector<InputTransformer>();

  for (auto gateInfo : info.asReader().getInputs()) {
    InputTransformer transformer;
    if (gateInfo.getTypeInfo().hasIndex()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getIndexInputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasPlaintext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getPlaintextInputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasLweCiphertext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getLweCiphertextInputTransformer(
                      keyset, gateInfo, csprng, useSimulation));
    } else {
      return StringError("Malformed input gate info.");
    }
    inputTransformers.push_back(transformer);
  }
  return ValueExporter(info, inputTransformers);
};

Result<TransportValue> ValueExporter::prepareInput(Value arg, size_t pos) {
  if (pos >= inputTransformers.size()) {
    return StringError("Tried to prepare a Value for incorrect position.");
  }
  return inputTransformers[pos](arg);
}

Result<ValueDecrypter>
ValueDecrypter::create(const Message<concreteprotocol::CircuitInfo> &info,
                       const ClientKeyset &keyset, bool useSimulation) {
  auto outputTransformers = std::vector<OutputTransformer>();

  for (auto gateInfo : info.asReader().getOutputs()) {
    OutputTransformer transformer;
    if (gateInfo.getTypeInfo().hasIndex()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getIndexOutputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasPlaintext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getPlaintextOutputTransformer(gateInfo));
    } else if (gateInfo.getTypeInfo().hasLweCiphertext()) {
      OUTCOME_TRY(transformer,
                  TransformerFactory::getLweCiphertextOutputTransformer(
                      keyset, gateInfo, useSimulation));
    } else {
      return StringError("Malformed output gate info.");
    }
    outputTransformers.push_back(transformer);
  }

  return ValueDecrypter(outputTransformers);
}

Result<Value> ValueDecrypter::processOutput(TransportValue result, size_t pos) {
  if (pos >= outputTransformers.size()) {
    return StringError(
        "Tried to process a TransportValue for incorrect position.");
  }
  return outputTransformers[pos](result);
}

Result<ClientProgram>
ClientProgram::create(const Message<concreteprotocol::ProgramInfo> &info) {
  return ClientProgram(info);
}

Result<ValueExporter> ClientProgram::getValueExporter(
    std::string circuitName, const ClientKeyset &keyset,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng, bool useSimulation) {
  auto circuits = info.asReader().getCircuits();
  auto circuitInfo = circuits.begin();
  if (circuitInfo == circuits.end()) {
    return StringError("Tried to get unknown client circuit: `" + circuitName +
                       "`");
  }
  return ValueExporter::create(*circuitInfo, keyset, csprng, useSimulation);
}

Result<ValueDecrypter> ClientProgram::getValueDecrypter(
    std::string circuitName, const ClientKeyset &keyset,
    std::shared_ptr<csprng::EncryptionCSPRNG> csprng, bool useSimulation) {
  auto circuits = info.asReader().getCircuits();
  auto circuitInfo = circuits.begin();
  if (circuitInfo == circuits.end()) {
    return StringError("Tried to get unknown client circuit: `" + circuitName +
                       "`");
  }
  return ValueDecrypter::create(*circuitInfo, keyset, useSimulation);
}

} // namespace clientlib
} // namespace concretelang

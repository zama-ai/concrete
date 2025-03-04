// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <fstream>
#include <vector>

#include "concrete-cpu.h"
#include "concrete-protocol.capnp.h"
#include "concretelang/Support/CompilationFeedback.h"

using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {

void CircuitCompilationFeedback::fillFromCircuitInfo(
    concreteprotocol::CircuitInfo::Reader circuitInfo) {
  auto computeGateSize = [&](const concreteprotocol::GateInfo::Reader reader) {
    unsigned int nElements = 1;
    for (auto dimension : reader.getRawInfo().getShape().getDimensions()) {
      nElements *= dimension;
    }
    unsigned int gateScalarSize = reader.getRawInfo().getIntegerPrecision() / 8;
    return nElements * gateScalarSize;
  };
  // Compute the size of inputs
  totalInputsSize = 0;
  for (auto gateInfo : circuitInfo.getInputs()) {
    totalInputsSize += computeGateSize(gateInfo);
  }
  // Compute the size of outputs
  totalOutputsSize = 0;
  for (auto gateInfo : circuitInfo.getOutputs()) {
    totalOutputsSize += computeGateSize(gateInfo);
  }
  // Extract CRT decomposition
  crtDecompositionsOfOutputs = {};
  for (auto gate : circuitInfo.getOutputs()) {
    if (gate.getTypeInfo().hasLweCiphertext() &&
        gate.getTypeInfo().getLweCiphertext().getEncoding().hasInteger()) {
      auto integerEncoding =
          gate.getTypeInfo().getLweCiphertext().getEncoding().getInteger();
      if (integerEncoding.getMode().hasCrt()) {
        auto moduli = integerEncoding.getMode().getCrt().getModuli();
        std::vector<int64_t> moduliVector(moduli.size());
        for (size_t i = 0; i < moduli.size(); i++) {
          moduliVector[i] = moduli[i];
        }
        crtDecompositionsOfOutputs.push_back(moduliVector);
      } else {
        crtDecompositionsOfOutputs.push_back(std::vector<int64_t>{});
      }
    }
  }
  // Sets name
  name = circuitInfo.getName().cStr();
}

void ProgramCompilationFeedback::fillFromProgramInfo(
    const Message<concreteprotocol::ProgramInfo> &programInfo) {
  auto params = programInfo.asReader();

  // Compute the size of secret keys
  totalSecretKeysSize = 0;
  for (auto skInfo : params.getKeyset().getLweSecretKeys()) {
    assert(skInfo.getParams().getIntegerPrecision() % 8 == 0);
    auto byteSize = skInfo.getParams().getIntegerPrecision() / 8;
    totalSecretKeysSize +=
        (uint64_t)skInfo.getParams().getLweDimension() * byteSize;
  }
  // Compute the boostrap keys size
  totalBootstrapKeysSize = 0;
  for (auto bskInfo : params.getKeyset().getLweBootstrapKeys()) {
    assert(bskInfo.getInputId() <
           (uint32_t)params.getKeyset().getLweSecretKeys().size());
    auto inputKeyInfo =
        params.getKeyset().getLweSecretKeys()[bskInfo.getInputId()];
    assert(bskInfo.getOutputId() <
           (uint32_t)params.getKeyset().getLweSecretKeys().size());
    auto outputKeyInfo =
        params.getKeyset().getLweSecretKeys()[bskInfo.getOutputId()];
    assert(bskInfo.getParams().getIntegerPrecision() % 8 == 0);
    auto byteSize = bskInfo.getParams().getIntegerPrecision() / 8;
    auto inputLweDimension = inputKeyInfo.getParams().getLweDimension();
    auto outputLweDimension = outputKeyInfo.getParams().getLweDimension();
    auto level = bskInfo.getParams().getLevelCount();
    auto glweDimension = bskInfo.getParams().getGlweDimension();
    totalBootstrapKeysSize +=
        concrete_cpu_bootstrap_key_size_u64(
            level, glweDimension, outputLweDimension, inputLweDimension) *
        byteSize;
  }
  // Compute the keyswitch keys size
  totalKeyswitchKeysSize = 0;
  for (auto kskInfo : params.getKeyset().getLweKeyswitchKeys()) {
    assert(kskInfo.getInputId() <
           (uint32_t)params.getKeyset().getLweSecretKeys().size());
    auto inputKeyInfo =
        params.getKeyset().getLweSecretKeys()[kskInfo.getInputId()];
    assert(kskInfo.getOutputId() <
           (uint32_t)params.getKeyset().getLweSecretKeys().size());
    auto outputKeyInfo =
        params.getKeyset().getLweSecretKeys()[kskInfo.getOutputId()];
    assert(kskInfo.getParams().getIntegerPrecision() % 8 == 0);
    auto byteSize = kskInfo.getParams().getIntegerPrecision() / 8;
    auto inputLweDimension = inputKeyInfo.getParams().getLweDimension();
    auto outputLweDimension = outputKeyInfo.getParams().getLweDimension();
    auto level = kskInfo.getParams().getLevelCount();
    totalKeyswitchKeysSize +=
        concrete_cpu_keyswitch_key_size_u64(level, inputLweDimension,
                                            outputLweDimension) *
        byteSize;
  }
  // Compute the circuit feedbacks
  for (auto circuitInfo : params.getCircuits()) {
    CircuitCompilationFeedback feedback;
    feedback.fillFromCircuitInfo(circuitInfo);
    circuitFeedbacks.push_back(feedback);
  }
}

outcome::checked<ProgramCompilationFeedback, StringError>
ProgramCompilationFeedback::load(std::string jsonPath) {
  std::ifstream file(jsonPath);
  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));
  if (file.fail()) {
    return StringError("Cannot read file: ") << jsonPath;
  }
  auto expectedCompFeedback =
      llvm::json::parse<ProgramCompilationFeedback>(content);
  if (auto err = expectedCompFeedback.takeError()) {
    return StringError("Cannot open compilation feedback: ")
           << llvm::toString(std::move(err)) << "\n"
           << content << "\n";
  }
  return expectedCompFeedback.get();
}

llvm::json::Object memoryUsageToJson(
    const std::map<std::string, std::optional<int64_t>> &memoryUsagePerLoc) {
  auto object = llvm::json::Object();
  for (auto key : memoryUsagePerLoc) {
    object.insert({key.first, key.second});
  }

  return object;
}

llvm::json::Object statisticToJson(const Statistic &statistic) {
  auto object = llvm::json::Object();
  object.insert({"location", statistic.location});
  object.insert({"count", statistic.count});
  switch (statistic.operation) {
  case PrimitiveOperation::PBS:
    object.insert({"operation", "PBS"});
    break;
  case PrimitiveOperation::WOP_PBS:
    object.insert({"operation", "WOP_PBS"});
    break;
  case PrimitiveOperation::KEY_SWITCH:
    object.insert({"operation", "KEY_SWITCH"});
    break;
  case PrimitiveOperation::CLEAR_ADDITION:
    object.insert({"operation", "CLEAR_ADDITION"});
    break;
  case PrimitiveOperation::ENCRYPTED_ADDITION:
    object.insert({"operation", "ENCRYPTED_ADDITION"});
    break;
  case PrimitiveOperation::CLEAR_MULTIPLICATION:
    object.insert({"operation", "CLEAR_MULTIPLICATION"});
    break;
  case PrimitiveOperation::ENCRYPTED_NEGATION:
    object.insert({"operation", "ENCRYPTED_NEGATION"});
    break;
  }
  auto keysJson = llvm::json::Array();
  for (auto &key : statistic.keys) {
    KeyType type = key.first;
    size_t index = key.second;

    auto keyJson = llvm::json::Array();
    switch (type) {
    case KeyType::SECRET:
      keyJson.push_back("SECRET");
      break;
    case KeyType::BOOTSTRAP:
      keyJson.push_back("BOOTSTRAP");
      break;
    case KeyType::KEY_SWITCH:
      keyJson.push_back("KEY_SWITCH");
      break;
    case KeyType::PACKING_KEY_SWITCH:
      keyJson.push_back("PACKING_KEY_SWITCH");
      break;
    }
    keyJson.push_back((int64_t)index);

    keysJson.push_back(std::move(keyJson));
  }
  object.insert({"keys", std::move(keysJson)});
  return object;
}

llvm::json::Array statisticsToJson(const std::vector<Statistic> &statistics) {
  auto object = llvm::json::Array();
  for (auto statistic : statistics) {
    object.push_back(statisticToJson(statistic));
  }
  return object;
}

llvm::json::Array crtDecompositionToJson(
    const std::vector<std::vector<int64_t>> &crtDecompositionsOfOutputs) {
  auto object = llvm::json::Array();
  for (auto crtDec : crtDecompositionsOfOutputs) {
    auto inner = llvm::json::Array();
    for (auto val : crtDec) {
      inner.push_back(val);
    }
    object.push_back(std::move(inner));
  }
  return object;
}

llvm::json::Array circuitFeedbacksToJson(
    const std::vector<CircuitCompilationFeedback> &circuitFeedbacks) {
  auto object = llvm::json::Array();
  for (auto circuit : circuitFeedbacks) {
    llvm::json::Object circuitObject{
        {"name", circuit.name},
        {"totalInputsSize", circuit.totalInputsSize},
        {"totalOutputsSize", circuit.totalOutputsSize},
        {"crtDecompositionsOfOutputs",
         crtDecompositionToJson(circuit.crtDecompositionsOfOutputs)},
        {"statistics", statisticsToJson(circuit.statistics)},
        {"memoryUsagePerLoc", memoryUsageToJson(circuit.memoryUsagePerLoc)},
    };
    object.push_back(std::move(circuitObject));
  }
  return object;
}

llvm::json::Value
toJSON(const mlir::concretelang::ProgramCompilationFeedback &program) {
  llvm::json::Object programObject{
      {"complexity", program.complexity},
      {"pError", program.pError},
      {"globalPError", program.globalPError},
      {"totalSecretKeysSize", program.totalSecretKeysSize},
      {"totalBootstrapKeysSize", program.totalBootstrapKeysSize},
      {"totalKeyswitchKeysSize", program.totalKeyswitchKeysSize},
      {"circuitFeedbacks", circuitFeedbacksToJson(program.circuitFeedbacks)}};
  return programObject;
}

template <typename K, typename V>
bool fromJSON(const llvm::json::Value &j, std::pair<K, V> &v,
              llvm::json::Path p) {
  if (auto *array = j.getAsArray()) {
    if (!fromJSON((*array)[0], v.first, p.index(0)))
      return false;
    if (!fromJSON((*array)[1], v.second, p.index(1)))
      return false;
    return true;
  }
  p.report("expected array");
  return false;
}

bool fromJSON(const llvm::json::Value j,
              mlir::concretelang::PrimitiveOperation &v, llvm::json::Path p) {
  if (auto operationString = j.getAsString()) {
    if (operationString == "PBS") {
      v = PrimitiveOperation::PBS;
      return true;
    } else if (operationString == "KEY_SWITCH") {
      v = PrimitiveOperation::KEY_SWITCH;
      return true;
    } else if (operationString == "WOP_PBS") {
      v = PrimitiveOperation::WOP_PBS;
      return true;
    } else if (operationString == "CLEAR_ADDITION") {
      v = PrimitiveOperation::CLEAR_ADDITION;
      return true;
    } else if (operationString == "ENCRYPTED_ADDITION") {
      v = PrimitiveOperation::ENCRYPTED_ADDITION;
      return true;
    } else if (operationString == "CLEAR_MULTIPLICATION") {
      v = PrimitiveOperation::CLEAR_MULTIPLICATION;
      return true;
    } else if (operationString == "ENCRYPTED_NEGATION") {
      v = PrimitiveOperation::ENCRYPTED_NEGATION;
      return true;
    } else {
      p.report("expected one of "
               "(PBS|KEY_SWITCH|WOP_PBS|CLEAR_ADDITION|ENCRYPTED_ADDITION|"
               "CLEAR_MULTIPLICATION|ENCRYPTED_NEGATION)");
      return false;
    }
  }
  p.report("expected string");
  return false;
}

bool fromJSON(const llvm::json::Value j, mlir::concretelang::KeyType &v,
              llvm::json::Path p) {
  if (auto keyTypeString = j.getAsString()) {
    if (keyTypeString == "SECRET") {
      v = KeyType::SECRET;
      return true;
    } else if (keyTypeString == "BOOTSTRAP") {
      v = KeyType::BOOTSTRAP;
      return true;
    } else if (keyTypeString == "KEY_SWITCH") {
      v = KeyType::KEY_SWITCH;
      return true;
    } else if (keyTypeString == "PACKING_KEY_SWITCH") {
      v = KeyType::PACKING_KEY_SWITCH;
      return true;
    } else {
      p.report(
          "expected one of (SECRET|BOOTSTRAP|KEY_SWITCH|PACKING_KEY_SWITCH)");
      return false;
    }
  }
  p.report("expected string");
  return false;
}

bool fromJSON(const llvm::json::Value j, mlir::concretelang::Statistic &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);

  return O && O.map("location", v.location) &&
         O.map("operation", v.operation) && O.map("operation", v.operation) &&
         O.map("keys", v.keys) && O.map("count", v.count);
}

bool fromJSON(const llvm::json::Value j,
              mlir::concretelang::CircuitCompilationFeedback &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("name", v.name) &&
         O.map("totalInputsSize", v.totalInputsSize) &&
         O.map("totalOutputsSize", v.totalOutputsSize) &&
         O.map("crtDecompositionsOfOutputs", v.crtDecompositionsOfOutputs) &&
         O.map("statistics", v.statistics) &&
         O.map("memoryUsagePerLoc", v.memoryUsagePerLoc);
}

bool fromJSON(const llvm::json::Value j,
              mlir::concretelang::ProgramCompilationFeedback &v,
              llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);

  return O && O.map("complexity", v.complexity) && O.map("pError", v.pError) &&
         O.map("globalPError", v.globalPError) &&
         O.map("totalSecretKeysSize", v.totalSecretKeysSize) &&
         O.map("totalBootstrapKeysSize", v.totalBootstrapKeysSize) &&
         O.map("totalKeyswitchKeysSize", v.totalKeyswitchKeysSize) &&
         O.map("circuitFeedbacks", v.circuitFeedbacks);
}

} // namespace concretelang
} // namespace mlir

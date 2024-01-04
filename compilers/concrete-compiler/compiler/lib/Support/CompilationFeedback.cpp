// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <fstream>

#include "concretelang/Support/CompilationFeedback.h"

using concretelang::protocol::Message;

namespace mlir {
namespace concretelang {

void CompilationFeedback::fillFromProgramInfo(
    const Message<concreteprotocol::ProgramInfo> &programInfo) {
  auto params = programInfo.asReader();

  // Compute the size of secret keys
  totalSecretKeysSize = 0;
  for (auto skInfo : params.getKeyset().getLweSecretKeys()) {
    assert(skInfo.getParams().getIntegerPrecision() % 8 == 0);
    auto byteSize = skInfo.getParams().getIntegerPrecision() / 8;
    totalSecretKeysSize += skInfo.getParams().getLweDimension() * byteSize;
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
    auto byteSize = bskInfo.getParams().getIntegerPrecision() % 8;
    auto inputLweSize = inputKeyInfo.getParams().getLweDimension() + 1;
    auto outputLweSize = outputKeyInfo.getParams().getLweDimension() + 1;
    auto level = bskInfo.getParams().getLevelCount();
    auto glweDimension = bskInfo.getParams().getGlweDimension();
    totalBootstrapKeysSize += inputLweSize * level * (glweDimension + 1) *
                              (glweDimension + 1) * outputLweSize * byteSize;
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
    auto byteSize = kskInfo.getParams().getIntegerPrecision() % 8;
    auto inputLweSize = inputKeyInfo.getParams().getLweDimension() + 1;
    auto outputLweSize = outputKeyInfo.getParams().getLweDimension() + 1;
    auto level = kskInfo.getParams().getLevelCount();
    totalKeyswitchKeysSize += level * inputLweSize * outputLweSize * byteSize;
  }
  auto circuitInfo = params.getCircuits()[0];
  auto computeGateSize =
      [&](const Message<concreteprotocol::GateInfo> &gateInfo) {
        unsigned int nElements = 1;
        // TODO: CHANGE THAT ITS WRONG
        for (auto dimension :
             gateInfo.asReader().getRawInfo().getShape().getDimensions()) {
          nElements *= dimension;
        }
        unsigned int gateScalarSize =
            gateInfo.asReader().getRawInfo().getIntegerPrecision() / 8;
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
}

outcome::checked<CompilationFeedback, StringError>
CompilationFeedback::load(std::string jsonPath) {
  std::ifstream file(jsonPath);
  std::string content((std::istreambuf_iterator<char>(file)),
                      (std::istreambuf_iterator<char>()));
  if (file.fail()) {
    return StringError("Cannot read file: ") << jsonPath;
  }
  auto expectedCompFeedback = llvm::json::parse<CompilationFeedback>(content);
  if (auto err = expectedCompFeedback.takeError()) {
    return StringError("Cannot open compilation feedback: ")
           << llvm::toString(std::move(err)) << "\n"
           << content << "\n";
  }
  return expectedCompFeedback.get();
}

llvm::json::Value toJSON(const mlir::concretelang::CompilationFeedback &v) {
  llvm::json::Object object{
      {"complexity", v.complexity},
      {"pError", v.pError},
      {"globalPError", v.globalPError},
      {"totalSecretKeysSize", v.totalSecretKeysSize},
      {"totalBootstrapKeysSize", v.totalBootstrapKeysSize},
      {"totalKeyswitchKeysSize", v.totalKeyswitchKeysSize},
      {"totalInputsSize", v.totalInputsSize},
      {"totalOutputsSize", v.totalOutputsSize},
      {"crtDecompositionsOfOutputs", v.crtDecompositionsOfOutputs},
  };

  auto memoryUsageObject = llvm::json::Object();
  for (auto key : v.memoryUsagePerLoc) {
    memoryUsageObject.insert({key.first, key.second});
  }
  object.insert({"memoryUsagePerLoc", std::move(memoryUsageObject)});

  auto statisticsJson = llvm::json::Array();
  for (auto statistic : v.statistics) {
    auto statisticJson = llvm::json::Object();
    statisticJson.insert({"location", statistic.location});
    switch (statistic.operation) {
    case PrimitiveOperation::PBS:
      statisticJson.insert({"operation", "PBS"});
      break;
    case PrimitiveOperation::WOP_PBS:
      statisticJson.insert({"operation", "WOP_PBS"});
      break;
    case PrimitiveOperation::KEY_SWITCH:
      statisticJson.insert({"operation", "KEY_SWITCH"});
      break;
    case PrimitiveOperation::CLEAR_ADDITION:
      statisticJson.insert({"operation", "CLEAR_ADDITION"});
      break;
    case PrimitiveOperation::ENCRYPTED_ADDITION:
      statisticJson.insert({"operation", "ENCRYPTED_ADDITION"});
      break;
    case PrimitiveOperation::CLEAR_MULTIPLICATION:
      statisticJson.insert({"operation", "CLEAR_MULTIPLICATION"});
      break;
    case PrimitiveOperation::ENCRYPTED_NEGATION:
      statisticJson.insert({"operation", "ENCRYPTED_NEGATION"});
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
    statisticJson.insert({"keys", std::move(keysJson)});
    statisticJson.insert({"count", statistic.count});

    statisticsJson.push_back(std::move(statisticJson));
  }
  object.insert({"statistics", std::move(statisticsJson)});

  return object;
}

bool fromJSON(const llvm::json::Value j,
              mlir::concretelang::CompilationFeedback &v, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);

  bool is_success =
      O && O.map("complexity", v.complexity) && O.map("pError", v.pError) &&
      O.map("globalPError", v.globalPError) &&
      O.map("totalSecretKeysSize", v.totalSecretKeysSize) &&
      O.map("totalBootstrapKeysSize", v.totalBootstrapKeysSize) &&
      O.map("totalKeyswitchKeysSize", v.totalKeyswitchKeysSize) &&
      O.map("totalInputsSize", v.totalInputsSize) &&
      O.map("totalOutputsSize", v.totalOutputsSize) &&
      O.map("crtDecompositionsOfOutputs", v.crtDecompositionsOfOutputs);

  if (!is_success) {
    return false;
  }

  auto object = j.getAsObject();
  if (!object) {
    return false;
  }

  auto memoryUsageObject = object->getObject("memoryUsagePerLoc");
  if (!memoryUsageObject) {
    return false;
  }
  for (auto entry : *memoryUsageObject) {
    auto loc = entry.getFirst().str();
    auto maybeUsage = entry.getSecond().getAsInteger();
    if (!maybeUsage.has_value()) {
      return false;
    }
    v.memoryUsagePerLoc[loc] = *maybeUsage;
  }

  auto statistics = object->getArray("statistics");
  if (!statistics) {
    return false;
  }

  for (auto statisticValue : *statistics) {
    auto statistic = statisticValue.getAsObject();
    if (!statistic) {
      return false;
    }

    auto location = statistic->getString("location");
    auto operationStr = statistic->getString("operation");
    auto keysArray = statistic->getArray("keys");
    auto count = statistic->getInteger("count");

    if (!operationStr || !location || !keysArray || !count) {
      return false;
    }

    PrimitiveOperation operation;
    if (operationStr.value() == "PBS") {
      operation = PrimitiveOperation::PBS;
    } else if (operationStr.value() == "KEY_SWITCH") {
      operation = PrimitiveOperation::KEY_SWITCH;
    } else if (operationStr.value() == "WOP_PBS") {
      operation = PrimitiveOperation::WOP_PBS;
    } else if (operationStr.value() == "CLEAR_ADDITION") {
      operation = PrimitiveOperation::CLEAR_ADDITION;
    } else if (operationStr.value() == "ENCRYPTED_ADDITION") {
      operation = PrimitiveOperation::ENCRYPTED_ADDITION;
    } else if (operationStr.value() == "CLEAR_MULTIPLICATION") {
      operation = PrimitiveOperation::CLEAR_MULTIPLICATION;
    } else if (operationStr.value() == "ENCRYPTED_NEGATION") {
      operation = PrimitiveOperation::ENCRYPTED_NEGATION;
    } else {
      return false;
    }

    auto keys = std::vector<std::pair<KeyType, size_t>>();
    for (auto keyValue : *keysArray) {
      llvm::json::Array *keyArray = keyValue.getAsArray();
      if (!keyArray || keyArray->size() != 2) {
        return false;
      }

      auto typeStr = keyArray->front().getAsString();
      auto index = keyArray->back().getAsInteger();

      if (!typeStr || !index) {
        return false;
      }

      KeyType type;
      if (typeStr.value() == "SECRET") {
        type = KeyType::SECRET;
      } else if (typeStr.value() == "BOOTSTRAP") {
        type = KeyType::BOOTSTRAP;
      } else if (typeStr.value() == "KEY_SWITCH") {
        type = KeyType::KEY_SWITCH;
      } else if (typeStr.value() == "PACKING_KEY_SWITCH") {
        type = KeyType::PACKING_KEY_SWITCH;
      } else {
        return false;
      }

      keys.push_back(std::make_pair(type, (size_t)*index));
    }

    v.statistics.push_back(
        Statistic{location->str(), operation, keys, (uint64_t)*count});
  }

  return true;
}

} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <fstream>

#include "llvm/Support/JSON.h"

#include "concretelang/Support/CompilationFeedback.h"

namespace mlir {
namespace concretelang {

void CompilationFeedback::fillFromClientParameters(
    ::concretelang::clientlib::ClientParameters params) {
  // Compute the size of secret keys
  totalSecretKeysSize = 0;
  for (auto sk : params.secretKeys) {
    totalSecretKeysSize += sk.byteSize();
  }
  // Compute the boostrap keys size
  totalBootstrapKeysSize = 0;
  for (auto bskParam : params.bootstrapKeys) {
    assert(bskParam.inputSecretKeyID < params.secretKeys.size());
    auto inputKey = params.secretKeys[bskParam.inputSecretKeyID];

    assert(bskParam.outputSecretKeyID < params.secretKeys.size());
    auto outputKey = params.secretKeys[bskParam.outputSecretKeyID];

    totalBootstrapKeysSize +=
        bskParam.byteSize(inputKey.lweSize(), outputKey.lweSize());
  }
  // Compute the keyswitch keys size
  totalKeyswitchKeysSize = 0;
  for (auto kskParam : params.keyswitchKeys) {
    assert(kskParam.inputSecretKeyID < params.secretKeys.size());
    auto inputKey = params.secretKeys[kskParam.inputSecretKeyID];
    assert(kskParam.outputSecretKeyID < params.secretKeys.size());
    auto outputKey = params.secretKeys[kskParam.outputSecretKeyID];
    totalKeyswitchKeysSize +=
        kskParam.byteSize(inputKey.lweSize(), outputKey.lweSize());
  }
  // Compute the size of inputs
  totalInputsSize = 0;
  for (auto gate : params.inputs) {
    totalInputsSize += gate.byteSize(params.secretKeys);
  }
  // Compute the size of outputs
  totalOutputsSize = 0;
  for (auto gate : params.outputs) {
    totalOutputsSize += gate.byteSize(params.secretKeys);
  }
  // Extract CRT decomposition
  crtDecompositionsOfOutputs = {};
  for (auto gate : params.outputs) {
    std::vector<int64_t> decomposition;
    if (gate.encryption.has_value()) {
      decomposition = gate.encryption->encoding.crt;
    }
    crtDecompositionsOfOutputs.push_back(decomposition);
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
    statisticJson.insert({"count", (int64_t)statistic.count});

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

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <fstream>

#include "boost/outcome.h"
#include "llvm/Support/JSON.h"

#include "concretelang/Support/CompilationFeedback.h"

namespace mlir {
namespace concretelang {

void CompilationFeedback::fillFromClientParameters(
    ::concretelang::clientlib::ClientParameters params) {
  // Compute the size of secret keys
  totalSecretKeysSize = 0;
  for (auto sk : params.secretKeys) {
    totalSecretKeysSize += sk.second.byteSize();
  }
  // Compute the boostrap keys size
  totalBootstrapKeysSize = 0;
  for (auto bsk : params.bootstrapKeys) {
    auto bskParam = bsk.second;
    auto inputKey = params.secretKeys.find(bskParam.inputSecretKeyID);
    assert(inputKey != params.secretKeys.end());
    auto outputKey = params.secretKeys.find(bskParam.outputSecretKeyID);
    assert(outputKey != params.secretKeys.end());

    totalBootstrapKeysSize += bskParam.byteSize(inputKey->second.lweSize(),
                                                outputKey->second.lweSize());
  }
  // Compute the keyswitch keys size
  totalKeyswitchKeysSize = 0;
  for (auto ksk : params.keyswitchKeys) {
    auto kskParam = ksk.second;
    auto inputKey = params.secretKeys.find(kskParam.inputSecretKeyID);
    assert(inputKey != params.secretKeys.end());
    auto outputKey = params.secretKeys.find(kskParam.outputSecretKeyID);
    assert(outputKey != params.secretKeys.end());
    totalKeyswitchKeysSize += kskParam.byteSize(inputKey->second.lweSize(),
                                                outputKey->second.lweSize());
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
    if (gate.encryption.hasValue()) {
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
    return StringError("Cannot open client parameters: ")
           << llvm::toString(std::move(err)) << "\n"
           << content << "\n";
  }
  return expectedCompFeedback.get();
}

llvm::json::Value toJSON(const mlir::concretelang::CompilationFeedback &v) {
  llvm::json::Object object{
      {"complexity", v.complexity},
      {"totalSecretKeysSize", v.totalSecretKeysSize},
      {"totalBootstrapKeysSize", v.totalBootstrapKeysSize},
      {"totalKeyswitchKeysSize", v.totalKeyswitchKeysSize},
      {"totalInputsSize", v.totalInputsSize},
      {"totalOutputsSize", v.totalOutputsSize},
      {"crtDecompositionsOfOutputs", v.crtDecompositionsOfOutputs},
  };
  return object;
}

bool fromJSON(const llvm::json::Value j,
              mlir::concretelang::CompilationFeedback &v, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("complexity", v.complexity) &&
         O.map("totalSecretKeysSize", v.totalSecretKeysSize) &&
         O.map("totalBootstrapKeysSize", v.totalBootstrapKeysSize) &&
         O.map("totalKeyswitchKeysSize", v.totalKeyswitchKeysSize) &&
         O.map("totalInputsSize", v.totalInputsSize) &&
         O.map("totalOutputsSize", v.totalOutputsSize) &&
         O.map("crtDecompositionsOfOutputs", v.crtDecompositionsOfOutputs);
}

} // namespace concretelang
} // namespace mlir

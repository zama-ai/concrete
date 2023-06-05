// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <cassert>
#include <fstream>

#include "boost/outcome.h"
#include "llvm/Support/JSON.h"

#include "concrete-protocol.pb.h"
#include "concretelang/Support/CompilationFeedback.h"

namespace mlir {
namespace concretelang {

inline size_t bitWidthAsWord(size_t exactBitWidth) {
  if (exactBitWidth <= 8)
    return 8;
  if (exactBitWidth <= 16)
    return 16;
  if (exactBitWidth <= 32)
    return 32;
  if (exactBitWidth <= 64)
    return 64;
  assert(false && "Bit witdh > 64 not supported");
}

void CompilationFeedback::fillFromProgramInfo(
    concreteprotocol::ProgramInfo &params) {
  // Compute the size of secret keys
  totalSecretKeysSize = 0;
  for (auto skInfo : params.keyset().lwesecretkeys()) {
    assert(skInfo.params().integerprecision() % 8 == 0);
    auto byteSize = skInfo.params().integerprecision() / 8;
    totalSecretKeysSize += skInfo.params().lwedimension() * byteSize;
  }
  // Compute the boostrap keys size
  totalBootstrapKeysSize = 0;
  for (auto bskInfo : params.keyset().lwebootstrapkeys()) {
    assert(bskInfo.inputid() < (uint32_t)params.keyset().lwesecretkeys_size());
    auto inputKeyInfo = params.keyset().lwesecretkeys(bskInfo.inputid());
    assert(bskInfo.outputid() < (uint32_t)params.keyset().lwesecretkeys_size());
    auto outputKeyInfo = params.keyset().lwesecretkeys(bskInfo.outputid());
    assert(bskInfo.params().integerprecision() % 8 == 0);
    auto byteSize = bskInfo.params().integerprecision() % 8;
    auto inputLweSize = inputKeyInfo.params().lwedimension() + 1;
    auto outputLweSize = outputKeyInfo.params().lwedimension() + 1;
    auto level = bskInfo.params().levelcount();
    auto glweDimension = bskInfo.params().glwedimension();
    totalBootstrapKeysSize += inputLweSize * level * (glweDimension + 1) *
                              (glweDimension + 1) * outputLweSize * byteSize;
  }
  // Compute the keyswitch keys size
  totalKeyswitchKeysSize = 0;
  for (auto kskInfo : params.keyset().lwekeyswitchkeys()) {
    assert(kskInfo.inputid() < (uint32_t)params.keyset().lwesecretkeys_size());
    auto inputKeyInfo = params.keyset().lwesecretkeys(kskInfo.inputid());
    assert(kskInfo.outputid() < (uint32_t)params.keyset().lwesecretkeys_size());
    auto outputKeyInfo = params.keyset().lwesecretkeys(kskInfo.outputid());
    assert(kskInfo.params().integerprecision() % 8 == 0);
    auto byteSize = kskInfo.params().integerprecision() % 8;
    auto inputLweSize = inputKeyInfo.params().lwedimension() + 1;
    auto outputLweSize = outputKeyInfo.params().lwedimension() + 1;
    auto level = kskInfo.params().levelcount();
    totalKeyswitchKeysSize += level * inputLweSize * outputLweSize * byteSize;
  }
  auto circuitInfo = params.circuits(0);
  auto computeGateSize = [&](concreteprotocol::GateInfo &gateInfo) {
    unsigned int nElements = 1;
    for (auto dimension : gateInfo.shape().dimensions()) {
      nElements *= dimension;
    }
    unsigned int gateScalarSize = 0;
    if (gateInfo.has_plaintext()) {
      auto plaintextGate = gateInfo.plaintext();
      gateScalarSize = bitWidthAsWord(plaintextGate.integerprecision()) / 8;
    } else if (gateInfo.has_index()) {
      auto indexGate = gateInfo.index();
      gateScalarSize = bitWidthAsWord(indexGate.integerprecision()) / 8;
    } else if (gateInfo.has_lweciphertext()) {
      auto lweCiphertextGate = gateInfo.lweciphertext();
      auto lweCiphertextGateEncryption = lweCiphertextGate.encryption();
      assert(lweCiphertextGateEncryption.integerprecision() % 8 == 0);
      if (lweCiphertextGate.has_integer()) {
        auto integerEncoding = lweCiphertextGate.integer();
        auto lweSize = lweCiphertextGateEncryption.lwedimension() + 1;
        auto byteSize = lweCiphertextGateEncryption.integerprecision() / 8;
        gateScalarSize = lweSize * byteSize;
        if (integerEncoding.has_native()) {
          // gateScalarSize does not need multiplier
        } else if (integerEncoding.has_chunked()) {
          nElements *= integerEncoding.chunked().size();
        } else if (integerEncoding.has_crt()) {
          nElements *= integerEncoding.crt().moduli_size();
        }
      } else if (lweCiphertextGate.has_boolean()) {
        // gateScalarSize does not need multiplier
      } else {
        assert(false);
      }
    } else {
      assert(false);
    }
    return nElements * gateScalarSize;
  };
  // Compute the size of inputs
  totalInputsSize = 0;
  for (auto gateInfo : circuitInfo.inputs()) {
    totalInputsSize += computeGateSize(gateInfo);
  }
  // Compute the size of outputs
  totalOutputsSize = 0;
  for (auto gateInfo : circuitInfo.outputs()) {
    totalOutputsSize += computeGateSize(gateInfo);
  }
  // Extract CRT decomposition
  crtDecompositionsOfOutputs = {};
  for (auto gate : circuitInfo.outputs()) {
    if (gate.has_lweciphertext() && gate.lweciphertext().has_integer()) {
      auto integerEncoding = gate.lweciphertext().integer();
      if (integerEncoding.has_crt()) {
        auto moduli = integerEncoding.crt().moduli();
        crtDecompositionsOfOutputs.push_back(
            std::vector<int64_t>(moduli.begin(), moduli.end()));
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
    return StringError("Cannot open client parameters: ")
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
  return object;
}

bool fromJSON(const llvm::json::Value j,
              mlir::concretelang::CompilationFeedback &v, llvm::json::Path p) {
  llvm::json::ObjectMapper O(j, p);
  return O && O.map("complexity", v.complexity) && O.map("pError", v.pError) &&
         O.map("globalPError", v.globalPError) &&
         O.map("totalSecretKeysSize", v.totalSecretKeysSize) &&
         O.map("totalBootstrapKeysSize", v.totalBootstrapKeysSize) &&
         O.map("totalKeyswitchKeysSize", v.totalKeyswitchKeysSize) &&
         O.map("totalInputsSize", v.totalInputsSize) &&
         O.map("totalOutputsSize", v.totalOutputsSize) &&
         O.map("crtDecompositionsOfOutputs", v.crtDecompositionsOfOutputs);
}

} // namespace concretelang
} // namespace mlir

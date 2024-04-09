// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Support/TFHECircuitKeys.h"
#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "llvm/ADT/SmallVector.h"
#include <llvm/ADT/SmallVector.h>
#include <optional>

// Faster than using a full fledged hash-set for small sets, and the array can
// be recovered right away.
template <typename V> struct SmallSet {
  llvm::SmallVector<V, 10> vector;

  void insert(V val) {
    for (auto vectorVal : vector) {
      if (vectorVal == val) {
        return;
      }
    }
    vector.push_back(val);
  }
};

template <typename V, unsigned N>
std::optional<size_t> vectorIndex(llvm::SmallVector<V, N> vector, V val) {
  for (size_t i = 0; i < vector.size(); i++) {
    auto potentialVal = vector[i];
    if (potentialVal == val) {
      return i;
    }
  }
  return std::nullopt;
}

namespace mlir {
namespace concretelang {
namespace TFHE {

template <typename V, unsigned int N>
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const mlir::SmallVector<V, N> vect) {
  OS << "[";
  for (auto v : vect) {
    OS << v << ",";
  }
  OS << "]";
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const TFHECircuitKeys cks) {

  OS << "TFHECircuitKeys{\n"
     << "    secretKeys:" << cks.secretKeys << "\n"
     << "    inputKeys:" << cks.inputKeys << "\n"
     << "    keyswitchKeys:" << cks.keyswitchKeys << "\n"
     << "    bootstrapKeys:" << cks.bootstrapKeys << "\n"
     << "    packingKeyswitchKeys:" << cks.packingKeyswitchKeys
     << "\n"
        "}";
  return OS;
}

TFHECircuitKeys extractCircuitKeys(mlir::ModuleOp moduleOp) {
  // Gathering circuit secret keys
  SmallSet<TFHE::GLWESecretKey> secretKeys;
  SmallSet<TFHE::GLWESecretKey> inputKeys;
  auto tryInsert = [&](mlir::Type type, bool input) {
    std::optional<TFHE::GLWESecretKey> key;
    if (auto glweType = type.dyn_cast<TFHE::GLWECipherTextType>()) {
      key.emplace(glweType.getKey());
    } else if (auto tensorType = type.dyn_cast<mlir::RankedTensorType>()) {
      if (auto elementType = tensorType.getElementType()
                                 .dyn_cast<TFHE::GLWECipherTextType>()) {
        key.emplace(elementType.getKey());
      }
    }
    if (key) {
      secretKeys.insert(key.value());
      if (input) {
        inputKeys.insert(key.value());
      }
    }
  };
  moduleOp->walk([&](mlir::Operation *op) {
    for (auto operand : op->getOperands()) {
      tryInsert(operand.getType(), false);
    }
    for (auto result : op->getResults()) {
      tryInsert(result.getType(), false);
    }
  });
  moduleOp->walk([&](mlir::func::FuncOp op) {
    for (auto argType : op.getArgumentTypes()) {
      tryInsert(argType, true);
    }
    for (auto resultType : op.getResultTypes()) {
      tryInsert(resultType, false);
    }
  });

  // Gathering circuit keyswitch keys
  SmallSet<TFHE::GLWEKeyswitchKeyAttr> keyswitchKeys;
  moduleOp->walk([&](TFHE::KeySwitchGLWEOp op) {
    keyswitchKeys.insert(op.getKeyAttr());
    secretKeys.insert(op.getKeyAttr().getInputKey());
    secretKeys.insert(op.getKeyAttr().getOutputKey());
  });

  moduleOp->walk([&](TFHE::BatchedKeySwitchGLWEOp op) {
    keyswitchKeys.insert(op.getKeyAttr());
    secretKeys.insert(op.getKeyAttr().getInputKey());
    secretKeys.insert(op.getKeyAttr().getOutputKey());
  });

  // Gathering circuit bootstrap keys
  SmallSet<TFHE::GLWEBootstrapKeyAttr> bootstrapKeys;
  moduleOp->walk([&](TFHE::BootstrapGLWEOp op) {
    bootstrapKeys.insert(op.getKeyAttr());
    secretKeys.insert(op.getKeyAttr().getInputKey());
    secretKeys.insert(op.getKeyAttr().getOutputKey());
  });

  // Gathering circuit packing keyswitch keys
  SmallSet<TFHE::GLWEPackingKeyswitchKeyAttr> packingKeyswitchKeys;
  moduleOp->walk([&](TFHE::WopPBSGLWEOp op) {
    keyswitchKeys.insert(op.getKskAttr());
    secretKeys.insert(op.getKskAttr().getInputKey());
    secretKeys.insert(op.getKskAttr().getOutputKey());
    bootstrapKeys.insert(op.getBskAttr());
    secretKeys.insert(op.getBskAttr().getInputKey());
    secretKeys.insert(op.getBskAttr().getOutputKey());
    packingKeyswitchKeys.insert(op.getPkskAttr());
    secretKeys.insert(op.getPkskAttr().getInputKey());
    secretKeys.insert(op.getPkskAttr().getOutputKey());
  });

  return TFHECircuitKeys{secretKeys.vector, inputKeys.vector,
                         bootstrapKeys.vector, keyswitchKeys.vector,
                         packingKeyswitchKeys.vector};
}

std::optional<uint64_t>
TFHE::TFHECircuitKeys::getSecretKeyIndex(TFHE::GLWESecretKey key) {
  return vectorIndex(this->secretKeys, key);
}

std::optional<uint64_t>
TFHE::TFHECircuitKeys::getBootstrapKeyIndex(TFHE::GLWEBootstrapKeyAttr key) {
  return vectorIndex(this->bootstrapKeys, key);
}

std::optional<uint64_t>
TFHE::TFHECircuitKeys::getKeyswitchKeyIndex(TFHE::GLWEKeyswitchKeyAttr key) {
  return vectorIndex(this->keyswitchKeys, key);
}

std::optional<uint64_t> TFHE::TFHECircuitKeys::getPackingKeyswitchKeyIndex(
    TFHE::GLWEPackingKeyswitchKeyAttr key) {
  return vectorIndex(this->packingKeyswitchKeys, key);
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir

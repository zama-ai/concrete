// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SUPPORT_TFHECIRCUITKEYS_H_
#define CONCRETELANG_SUPPORT_TFHECIRCUITKEYS_H_

#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"

namespace mlir {
namespace concretelang {
namespace TFHE {

struct TFHECircuitKeys {
  llvm::SmallVector<TFHE::GLWESecretKey, 10> secretKeys;
  llvm::SmallVector<TFHE::GLWESecretKey, 10> inputKeys;
  llvm::SmallVector<TFHE::GLWEBootstrapKeyAttr, 10> bootstrapKeys;
  llvm::SmallVector<TFHE::GLWEKeyswitchKeyAttr, 10> keyswitchKeys;
  llvm::SmallVector<TFHE::GLWEPackingKeyswitchKeyAttr, 10> packingKeyswitchKeys;

  std::optional<uint64_t> getSecretKeyIndex(TFHE::GLWESecretKey key);
  std::optional<uint64_t> getKeyswitchKeyIndex(TFHE::GLWEKeyswitchKeyAttr key);
  std::optional<uint64_t> getBootstrapKeyIndex(TFHE::GLWEBootstrapKeyAttr key);
  std::optional<uint64_t>
  getPackingKeyswitchKeyIndex(TFHE::GLWEPackingKeyswitchKeyAttr key);
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const TFHECircuitKeys cks);

TFHECircuitKeys extractCircuitKeys(mlir::ModuleOp moduleOp);

} // namespace TFHE
} // namespace concretelang
} // namespace mlir
#endif

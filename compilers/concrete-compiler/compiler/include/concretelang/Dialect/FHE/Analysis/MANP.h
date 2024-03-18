// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_FHE_ANALYSIS_MANP_H
#define CONCRETELANG_DIALECT_FHE_ANALYSIS_MANP_H

#include <functional>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace concretelang {
bool isEncryptedValue(mlir::Value value);
unsigned int getEintPrecision(mlir::Value value);
std::unique_ptr<mlir::Pass> createMANPPass(bool debug = false);

std::unique_ptr<mlir::Pass>
createMaxMANPPass(std::function<void(uint64_t, unsigned)> setMax);
} // namespace concretelang
} // namespace mlir

#endif

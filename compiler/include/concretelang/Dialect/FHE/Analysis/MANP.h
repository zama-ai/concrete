// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#ifndef CONCRETELANG_DIALECT_FHE_ANALYSIS_MANP_H
#define CONCRETELANG_DIALECT_FHE_ANALYSIS_MANP_H

#include <functional>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::Pass> createMANPPass(bool debug = false);

std::unique_ptr<mlir::Pass>
createMaxMANPPass(std::function<void(const llvm::APInt &, unsigned)> setMax);
} // namespace concretelang
} // namespace mlir

#endif

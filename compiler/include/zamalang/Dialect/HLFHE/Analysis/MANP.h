#ifndef ZAMALANG_DIALECT_HLFHE_ANALYSIS_MANP_H
#define ZAMALANG_DIALECT_HLFHE_ANALYSIS_MANP_H

#include <functional>
#include <mlir/Pass/Pass.h>

namespace mlir {
namespace zamalang {
std::unique_ptr<mlir::Pass> createMANPPass(bool debug = false);

std::unique_ptr<mlir::Pass>
createMaxMANPPass(std::function<void(const llvm::APInt &, unsigned)> setMax);
} // namespace zamalang
} // namespace mlir

#endif

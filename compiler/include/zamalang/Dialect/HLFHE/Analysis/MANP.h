#ifndef ZAMALANG_DIALECT_HLFHE_ANALYSIS_MANP_H
#define ZAMALANG_DIALECT_HLFHE_ANALYSIS_MANP_H

#include <mlir/Pass/Pass.h>

namespace mlir {
namespace zamalang {
std::unique_ptr<mlir::Pass> createMANPPass(bool debug = false);
} // namespace zamalang
} // namespace mlir

#endif

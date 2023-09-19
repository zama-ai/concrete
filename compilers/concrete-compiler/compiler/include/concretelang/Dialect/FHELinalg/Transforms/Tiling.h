// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_LINALG_TILING_PASS_H
#define CONCRETELANG_LINALG_TILING_PASS_H

#include <concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/FHELinalg/Transforms/Tiling.h.inc>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<>>
createFHELinalgTilingMarkerPass(llvm::ArrayRef<int64_t> tileSizes);

std::unique_ptr<mlir::OperationPass<>> createLinalgTilingPass();
} // namespace concretelang
} // namespace mlir

#endif

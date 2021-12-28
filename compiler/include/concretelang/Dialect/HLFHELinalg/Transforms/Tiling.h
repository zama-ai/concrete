// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_HLFHELINALG_TILING_PASS_H
#define CONCRETELANG_HLFHELINALG_TILING_PASS_H

#include <mlir/Pass/Pass.h>
#include <concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/HLFHELinalg/Transforms/Tiling.h.inc>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<>>
createHLFHELinalgTilingMarkerPass(llvm::ArrayRef<int64_t> tileSizes);

std::unique_ptr<mlir::OperationPass<>> createHLFHELinalgTilingPass();
} // namespace concretelang
} // namespace mlir

#endif

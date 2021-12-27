// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef ZAMALANG_HLFHELINALG_TILING_PASS_H
#define ZAMALANG_HLFHELINALG_TILING_PASS_H

#include <mlir/Pass/Pass.h>
#include <zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h>

#define GEN_PASS_CLASSES
#include <zamalang/Dialect/HLFHELinalg/Transforms/Tiling.h.inc>

namespace mlir {
namespace zamalang {
std::unique_ptr<mlir::OperationPass<>>
createHLFHELinalgTilingMarkerPass(llvm::ArrayRef<int64_t> tileSizes);

std::unique_ptr<mlir::OperationPass<>> createHLFHELinalgTilingPass();
} // namespace zamalang
} // namespace mlir

#endif

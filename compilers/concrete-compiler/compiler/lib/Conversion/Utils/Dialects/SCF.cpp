// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Conversion/Utils/Dialects/SCF.h"
#include "mlir/Transforms/RegionUtils.h"

namespace mlir {
namespace concretelang {
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::ForOp, false>::matchAndRewrite(
    scf::ForOp oldOp, mlir::OpConversionPattern<scf::ForOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Create new for loop with empty body, but converted iter args
  scf::ForOp newForOp = rewriter.create<scf::ForOp>(
      oldOp.getLoc(), adaptor.getLowerBound(), adaptor.getUpperBound(),
      adaptor.getStep(), adaptor.getInitArgs(),
      [&](OpBuilder &builder, Location loc, Value iv, ValueRange args) {});

  newForOp->setAttrs(adaptor.getAttributes());

  // Move operations from old for op to new one
  auto &newOperations = newForOp.getBody()->getOperations();
  mlir::Block *oldBody = oldOp.getBody();

  newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                       oldBody->begin(), oldBody->end());

  // Remap iter args and IV
  for (auto argsPair : llvm::zip(oldOp.getBody()->getArguments(),
                                 newForOp.getBody()->getArguments())) {
    replaceAllUsesInRegionWith(std::get<0>(argsPair), std::get<1>(argsPair),
                               newForOp.getRegion());
  }

  rewriter.replaceOp(oldOp, newForOp.getResults());

  return mlir::success();
}
} // namespace concretelang
} // namespace mlir

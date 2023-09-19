// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
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

//
// Specializations for ForallOp
//
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::ForallOp, false>::matchAndRewrite(
    scf::ForallOp oldOp,
    mlir::OpConversionPattern<scf::ForallOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Create new forall operation with empty body, but converted iter
  // args
  llvm::SmallVector<mlir::OpFoldResult> lbs = getMixedValues(
      adaptor.getStaticLowerBound(), adaptor.getDynamicLowerBound(), rewriter);
  llvm::SmallVector<mlir::OpFoldResult> ubs = getMixedValues(
      adaptor.getStaticUpperBound(), adaptor.getDynamicUpperBound(), rewriter);
  llvm::SmallVector<mlir::OpFoldResult> step = getMixedValues(
      adaptor.getStaticStep(), adaptor.getDynamicStep(), rewriter);

  rewriter.setInsertionPoint(oldOp);

  scf::ForallOp newForallOp = rewriter.create<scf::ForallOp>(
      oldOp.getLoc(), lbs, ubs, step, adaptor.getOutputs(),
      adaptor.getMapping());

  newForallOp->setAttrs(adaptor.getAttributes());

  // Move operations from old for op to new one
  auto &newOperations = newForallOp.getBody()->getOperations();
  mlir::Block *oldBody = oldOp.getBody();

  newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                       oldBody->begin(), std::prev(oldBody->end()));

  // Move operations from `scf.forall.in_parallel` terminator of the
  // old op to the terminator of the new op

  mlir::scf::InParallelOp oldTerminator =
      llvm::dyn_cast<mlir::scf::InParallelOp>(*std::prev(oldBody->end()));

  assert(oldTerminator && "Last operation of `scf.forall` op expected be a "
                          "`scf.forall.in_parallel` op");

  mlir::scf::InParallelOp newTerminator = newForallOp.getTerminator();

  mlir::Block::OpListType &oldTerminatorOps =
      oldTerminator.getRegion().getBlocks().begin()->getOperations();
  mlir::Block::OpListType &newTerminatorOps =
      newTerminator.getRegion().getBlocks().begin()->getOperations();

  newTerminatorOps.splice(newTerminatorOps.begin(), oldTerminatorOps,
                          oldTerminatorOps.begin(), oldTerminatorOps.end());

  // Remap iter args and IV
  for (auto argsPair : llvm::zip(oldOp.getBody()->getArguments(),
                                 newForallOp.getBody()->getArguments())) {
    std::get<0>(argsPair).replaceAllUsesWith(std::get<1>(argsPair));
  }

  rewriter.replaceOp(oldOp, newForallOp.getResults());

  return mlir::success();
}

//
// Specializations for InParallelOp
//
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::InParallelOp, false>::matchAndRewrite(
    scf::InParallelOp oldOp,
    mlir::OpConversionPattern<scf::InParallelOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Create new for loop with empty body, but converted iter args
  scf::InParallelOp newInParallelOp =
      rewriter.replaceOpWithNewOp<scf::InParallelOp>(oldOp);

  // newInParallelOp->setAttrs(adaptor.getAttributes());

  // Move operations from old for op to new one
  auto &newOperations = newInParallelOp.getBody()->getOperations();
  mlir::Block *oldBody = oldOp.getBody();

  newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                       oldBody->begin(), oldBody->end());

  return mlir::success();
}

} // namespace concretelang
} // namespace mlir

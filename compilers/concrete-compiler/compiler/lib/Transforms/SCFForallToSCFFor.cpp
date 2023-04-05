// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Transforms/Passes.h"

namespace {
// Processes an array of OpFoldResults and returns the same array,
// but with all Values remapped using the provided IRMapping
llvm::SmallVector<mlir::OpFoldResult>
remapMixedOperands(llvm::ArrayRef<mlir::OpFoldResult> operands,
                   const mlir::IRMapping &mapping) {
  return llvm::to_vector(llvm::map_range(
      operands, [&](mlir::OpFoldResult v) -> mlir::OpFoldResult {
        return v.is<mlir::Value>()
                   ? mapping.lookupOrDefault(v.get<mlir::Value>())
                   : v;
      }));
}

struct SCFForallToSCFForPattern
    : public mlir::OpRewritePattern<mlir::scf::ForallOp> {
  SCFForallToSCFForPattern(::mlir::MLIRContext *context,
                           mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::scf::ForallOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForallOp forallOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(forallOp);

    mlir::Location loc = forallOp.getLoc();

    llvm::SmallVector<mlir::Value> lbs =
        mlir::getAsValues(rewriter, loc, forallOp.getMixedLowerBound());
    llvm::SmallVector<mlir::Value> ubs =
        mlir::getAsValues(rewriter, loc, forallOp.getMixedUpperBound());
    llvm::SmallVector<mlir::Value> steps =
        mlir::getAsValues(rewriter, loc, forallOp.getMixedStep());

    llvm::SmallVector<mlir::Value> iterArgs =
        llvm::to_vector_of<mlir::Value>(forallOp.getOutputs());

    mlir::ValueRange ivs = forallOp.getInductionVars();
    llvm::SmallVector<mlir::scf::ForOp> forOps;

    // Build an empty loop nest with the right bounds and iteration
    // arguments, propagating the iteration arguments inward
    mlir::IRMapping mapping;

    for (auto [lb, ub, step, iv] : llvm::zip_equal(lbs, ubs, steps, ivs)) {
      mlir::scf::ForOp forOp =
          rewriter.create<mlir::scf::ForOp>(loc, lb, ub, step, iterArgs);

      mlir::Block &body = *forOp.getLoopBody().getBlocks().begin();

      rewriter.setInsertionPoint(&body, body.begin());
      iterArgs = llvm::to_vector_of<mlir::Value>(forOp.getRegionIterArgs());

      mapping.map(iv, forOp.getInductionVar());
      forOps.push_back(forOp);
    }

    // Map the outputs of the original forall loop to the region
    // iteration arguments of the innermost loop
    for (auto [outArg, iterArg] :
         llvm::zip_equal(forallOp.getRegionOutArgs(), iterArgs)) {
      mapping.map(outArg, iterArg);
    }

    // Clone all operations of the original loop body, except the
    // scf.forall.in_parallel terminator
    mlir::Block *loopBodyBlock = forallOp.getBody();
    mlir::Block::iterator srcBlockEnd = std::prev(loopBodyBlock->end(), 2);

    for (auto it = loopBodyBlock->begin(); it != std::next(srcBlockEnd); it++) {
      rewriter.clone(*it, mapping);
    }

    // Rewrite the terminator, replacing all
    // `tensor.parallel_insert_slice` instances with equivalent
    // `tensor.insert_slice` ops.
    mlir::scf::InParallelOp ip = forallOp.getTerminator();

    // Handle scf.forall.in_parallel terminator
    for (auto it = ip.getBody()->begin(); it != ip.getBody()->end(); it++) {
      if (mlir::tensor::ParallelInsertSliceOp pis =
              llvm::dyn_cast<mlir::tensor::ParallelInsertSliceOp>(*it)) {
        mlir::Value updatedTensor =
            rewriter.create<mlir::tensor::InsertSliceOp>(
                pis.getLoc(), mapping.lookupOrDefault(pis.getSource()),
                mapping.lookupOrDefault(pis.getDest()),
                remapMixedOperands(pis.getMixedOffsets(), mapping),
                remapMixedOperands(pis.getMixedSizes(), mapping),
                remapMixedOperands(pis.getMixedStrides(), mapping));

        mapping.map(pis.getDest(), updatedTensor);

      } else {
        rewriter.clone(*it, mapping);
      }
    }

    // Create an `scf.yield` operation for each of the loops in the
    // loop nest, returning the updated tensors corresponding to the
    // output tensors of the forall operation for the innermost loop
    // and returning the produced values of the contained loop for the
    // outer loops.
    mlir::SmallVector<mlir::Value> retVals = llvm::to_vector(
        llvm::map_range(forallOp.getRegionOutArgs(), [&](mlir::Value v) {
          return mapping.lookupOrDefault(v);
        }));

    for (mlir::scf::ForOp forOp : llvm::reverse(forOps)) {
      rewriter.setInsertionPoint(forOp.getBody(), forOp.getBody()->end());
      rewriter.create<mlir::scf::YieldOp>(loc, retVals);

      retVals = forOp.getResults();
    }

    mlir::scf::ForOp outermostFor = forOps[0];
    rewriter.replaceOp(forallOp, outermostFor.getResults());

    return ::mlir::success();
  };
};

struct SCFForallToSCFForPass
    : public SCFForallToSCFForBase<SCFForallToSCFForPass> {

  SCFForallToSCFForPass() {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<SCFForallToSCFForPattern>(&getContext());

    // Mark ops from the target dialect as legal operations
    target.addIllegalOp<mlir::scf::ForallOp>();
    target.addIllegalOp<mlir::scf::InParallelOp>();
    target.addIllegalOp<mlir::tensor::ParallelInsertSliceOp>();

    // Mark all other ops as legal
    target.markUnknownOpDynamicallyLegal(
        [](mlir::Operation *) { return true; });

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createSCFForallToSCFForPass() {
  return std::make_unique<SCFForallToSCFForPass>();
}
} // namespace concretelang
} // namespace mlir

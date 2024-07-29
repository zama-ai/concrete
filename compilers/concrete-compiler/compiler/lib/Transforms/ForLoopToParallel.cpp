// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace {
class ForOpPattern : public mlir::OpRewritePattern<mlir::scf::ForOp> {
public:
  ForOpPattern(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::scf::ForOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp forOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto attr = forOp->getAttrOfType<mlir::BoolAttr>("parallel");

    if (!attr || !attr.getValue()) {
      return mlir::failure();
    }

    assert(forOp.getRegionIterArgs().size() == 0 &&
           "unexpecting iter args when loops are bufferized");

    rewriter.replaceOpWithNewOp<mlir::scf::ParallelOp>(
        forOp, mlir::ValueRange{forOp.getLowerBound()},
        mlir::ValueRange{forOp.getUpperBound()}, forOp.getStep(), std::nullopt,
        [&](mlir::OpBuilder &builder, mlir::Location location,
            mlir::ValueRange indVar, mlir::ValueRange iterArgs) {
          mlir::IRMapping map;
          map.map(forOp.getInductionVar(), indVar.front());
          for (auto &op : forOp.getRegion().front()) {
            auto newOp = builder.clone(op, map);
            map.map(op.getResults(), newOp->getResults());
          }
        });

    return mlir::success();
  }
};
} // namespace

namespace {
struct ForLoopToParallelPass
    : public ForLoopToParallelBase<ForLoopToParallelPass> {

  void runOnOperation() override {
    auto func = getOperation();
    auto *context = &getContext();
    mlir::RewritePatternSet patterns(context);
    mlir::ConversionTarget target(*context);
    patterns.add<ForOpPattern>(context);
    target.addDynamicallyLegalOp<mlir::scf::ForOp>([&](mlir::scf::ForOp op) {
      auto r = op->getAttrOfType<mlir::BoolAttr>("parallel") == nullptr;
      return r;
    });
    target.markUnknownOpDynamicallyLegal(
        [&](mlir::Operation *op) { return true; });
    if (mlir::applyPatternsAndFoldGreedily(func, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    };
  }
};
} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
mlir::concretelang::createForLoopToParallel() {
  return std::make_unique<ForLoopToParallelPass>();
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"

namespace {
struct LinalgFillToLinalgGenericPattern
    : public mlir::OpRewritePattern<mlir::linalg::FillOp> {
  LinalgFillToLinalgGenericPattern(::mlir::MLIRContext *context,
                                   mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::linalg::FillOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::linalg::FillOp fillOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    if (fillOp.getOutputs().size() != 1)
      return ::mlir::failure();

    mlir::RankedTensorType outputTensorType =
        fillOp.getOutputs()[0].getType().cast<mlir::RankedTensorType>();

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        outputTensorType.getRank(), mlir::utils::IteratorType::parallel);

    mlir::AffineMap map = mlir::AffineMap::getMultiDimIdentityMap(
        outputTensorType.getRank(), this->getContext());

    mlir::SmallVector<mlir::AffineMap> maps(1, map);

    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      nestedBuilder.create<mlir::linalg::YieldOp>(nestedLoc,
                                                  fillOp.getInputs()[0]);
    };

    rewriter.replaceOpWithNewOp<mlir::linalg::GenericOp>(
        fillOp, fillOp.getOutputs().getTypes(), mlir::ValueRange{},
        fillOp.getOutputs(), maps, iteratorTypes, bodyBuilder);

    return ::mlir::success();
  };
};

struct LinalgFillToLinalgGenericPass
    : public LinalgFillToLinalgGenericBase<LinalgFillToLinalgGenericPass> {

  LinalgFillToLinalgGenericPass() {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<LinalgFillToLinalgGenericPattern>(&getContext());

    target.addIllegalOp<mlir::linalg::FillOp>();
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
std::unique_ptr<OperationPass<ModuleOp>> createLinalgFillToLinalgGenericPass() {
  return std::make_unique<LinalgFillToLinalgGenericPass>();
}
} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/Transforms/Max/Max.h"

namespace arith = mlir::arith;
namespace func = mlir::func;

namespace FHE = mlir::concretelang::FHE;

/// This rewrite pattern transforms all instances
/// of `FHE.max_eint` to `max(x - y, 0) + y`.
struct MaxEintPattern : public mlir::OpRewritePattern<FHE::MaxEintOp> {
  MaxEintPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHE::MaxEintOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::MaxEintOp maxEintOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Get the operator indexes stored in FHE.mul operator to annotated FHE
    // nodes
    auto operatorIndexes =
        maxEintOp->getAttrOfType<mlir::DenseI32ArrayAttr>("TFHE.OId");

    const mlir::Location loc = maxEintOp->getLoc();

    const FHE::FheIntegerInterface outputTy =
        maxEintOp->getResult(0).getType().cast<FHE::FheIntegerInterface>();
    const int64_t outputBitWidth = outputTy.getWidth();

    mlir::Value x = maxEintOp.getX();
    mlir::Value y = maxEintOp.getY();

    const auto xTy = x.getType().cast<FHE::FheIntegerInterface>();
    const auto yTy = y.getType().cast<FHE::FheIntegerInterface>();

    const auto signedTy = FHE::EncryptedSignedIntegerType::get(
        this->getContext(), outputBitWidth);

    if (xTy.isUnsigned()) {
      x = rewriter.create<FHE::ToSignedOp>(loc, signedTy, x).getResult();
    }
    if (yTy.isUnsigned()) {
      y = rewriter.create<FHE::ToSignedOp>(loc, signedTy, y).getResult();
    }

    auto sub = rewriter.create<FHE::SubEintOp>(loc, x, y);
    if (operatorIndexes != nullptr)
      sub->setAttr("TFHE.OId", rewriter.getI32IntegerAttr(operatorIndexes[0]));

    const int64_t lutSize = 1 << outputBitWidth;

    auto lutValues = std::vector<int64_t>();
    for (int64_t i = 0; i < lutSize / 2; i++) {
      lutValues.push_back(i);
    }
    for (int64_t i = 0; i < lutSize / 2; i++) {
      lutValues.push_back(0);
    }

    const mlir::Attribute lutAttr = rewriter.getI64TensorAttr(lutValues);
    const mlir::Value lut =
        rewriter.create<arith::ConstantOp>(loc, lutAttr).getResult();

    auto max =
        rewriter.create<FHE::ApplyLookupTableEintOp>(loc, outputTy, sub, lut);
    if (operatorIndexes != nullptr) {
      // It's a signed lut and the signed correction can be saw as the same
      // optimizer node than the max sub
      max->setAttr("TFHE.OId",
                   rewriter.getDenseI32ArrayAttr(std::vector<int32_t>{
                       operatorIndexes[0], operatorIndexes[1]}));
    }

    auto add = rewriter.create<FHE::AddEintOp>(loc, max, maxEintOp.getY());
    if (operatorIndexes != nullptr)
      add->setAttr("TFHE.OId", rewriter.getI32IntegerAttr(operatorIndexes[2]));

    rewriter.replaceOp(maxEintOp, {add});
    return mlir::success();
  };
};

namespace {

struct FHEMaxTransform : public FHEMaxTransformBase<FHEMaxTransform> {
  void runOnOperation() final;
};

void FHEMaxTransform::runOnOperation() {
  auto target = mlir::ConversionTarget(this->getContext());
  target.addLegalDialect<arith::ArithDialect>();
  target.addLegalDialect<FHE::FHEDialect>();
  target.addIllegalOp<FHE::MaxEintOp>();

  auto patterns = mlir::RewritePatternSet(&this->getContext());
  patterns.insert<MaxEintPattern>(&this->getContext());

  mlir::Operation *op = this->getOperation();
  if (mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

} // namespace

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<>> createFHEMaxTransformPass() {
  return std::make_unique<FHEMaxTransform>();
}

} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Transforms/Passes.h"

namespace {
struct TensorEmptyToBufferizationAllocPattern
    : public mlir::OpRewritePattern<mlir::tensor::EmptyOp> {
  TensorEmptyToBufferizationAllocPattern(::mlir::MLIRContext *context,
                                         mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::EmptyOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::EmptyOp emptyOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::bufferization::AllocTensorOp>(
        emptyOp, emptyOp.getResult().getType().cast<mlir::RankedTensorType>(),
        mlir::ValueRange{});

    return ::mlir::success();
  };
};

struct TensorEmptyToBufferizationAllocPass
    : public TensorEmptyToBufferizationAllocBase<
          TensorEmptyToBufferizationAllocPass> {

  TensorEmptyToBufferizationAllocPass() {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<TensorEmptyToBufferizationAllocPattern>(&getContext());

    // Mark ops from the target dialect as legal operations
    target.addIllegalOp<mlir::tensor::EmptyOp>();

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
std::unique_ptr<OperationPass<ModuleOp>>
createTensorEmptyToBufferizationAllocPass() {
  return std::make_unique<TensorEmptyToBufferizationAllocPass>();
}
} // namespace concretelang
} // namespace mlir

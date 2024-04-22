// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <concretelang/Dialect/TFHE/IR/TFHEOps.h>
#include <concretelang/Dialect/TFHE/Transforms/Transforms.h>
#include <concretelang/Support/Constants.h>

namespace mlir {
namespace concretelang {

namespace {

class TensorGenerateToScfForall
    : public mlir::OpRewritePattern<mlir::tensor::GenerateOp> {
public:
  TensorGenerateToScfForall(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::tensor::GenerateOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::tensor::GenerateOp generateOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto resultingType = generateOp.getType().cast<mlir::RankedTensorType>();

    auto emptyTensorOp = rewriter.create<tensor::EmptyOp>(
        generateOp.getLoc(), resultingType, mlir::ValueRange{});

    auto upperBounds = llvm::SmallVector<mlir::OpFoldResult>();
    for (auto dimension : resultingType.getShape()) {
      upperBounds.push_back(
          mlir::OpFoldResult(rewriter.getIndexAttr(dimension)));
    }

    auto forallOp = rewriter.create<mlir::scf::ForallOp>(
        generateOp.getLoc(), upperBounds, mlir::ValueRange{emptyTensorOp},
        std::nullopt, nullptr);

    mlir::IRMapping mapping;
    for (auto [source, destination] :
         llvm::zip_equal(generateOp.getBody().getArguments(),
                         forallOp.getInductionVars())) {
      mapping.map(source, destination);
    }

    auto terminator = forallOp.getTerminator();
    rewriter.setInsertionPoint(terminator);

    auto &sourceBlock = generateOp.getBody().front();
    auto *destinationBlock = forallOp.getBody();

    auto yieldOp = sourceBlock.getTerminator();
    for (auto &op : sourceBlock) {
      if (&op != yieldOp) {
        rewriter.clone(op, mapping);
      }
    }
    auto yieldedValue = mapping.lookup(yieldOp->getOperand(0));

    auto element = yieldedValue;
    if (!yieldedValue.getType().isa<mlir::TensorType>()) {
      element = rewriter.create<mlir::tensor::FromElementsOp>(
          generateOp.getLoc(), element);
    }

    auto args = destinationBlock->getArguments();
    auto output = args.back();

    auto offsets = std::vector<mlir::OpFoldResult>();
    auto sizes = std::vector<mlir::OpFoldResult>();
    auto strides = std::vector<mlir::OpFoldResult>();

    for (size_t i = 0; i < args.size() - 1; i++) {
      offsets.push_back(mlir::OpFoldResult(args[i]));
      sizes.push_back(mlir::OpFoldResult(rewriter.getIndexAttr(1)));
      strides.push_back(mlir::OpFoldResult(rewriter.getIndexAttr(1)));
    }

    rewriter.setInsertionPointToStart(terminator.getBody());

    rewriter.create<mlir::tensor::ParallelInsertSliceOp>(
        generateOp.getLoc(), element, output, offsets, sizes, strides);

    rewriter.replaceOp(generateOp, forallOp.getResults());
    return mlir::success();
  }
};

class TFHEOperationTransformationsPass
    : public TFHEOperationTransformationsBase<
          TFHEOperationTransformationsPass> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());
    patterns.add<TensorGenerateToScfForall>(op->getContext());

    if (mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
      this->signalPassFailure();
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTFHEOperationTransformationsPass() {
  return std::make_unique<TFHEOperationTransformationsPass>();
}

} // namespace concretelang
} // namespace mlir

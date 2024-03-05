// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <concretelang/Dialect/TFHE/IR/TFHEOps.h>
#include <concretelang/Dialect/TFHE/Transforms/Transforms.h>
#include <concretelang/Support/Constants.h>

namespace mlir {
namespace concretelang {

namespace {

/// Get the constant integer that the cleartext was created from if it exists.
std::optional<IntegerAttr>
getConstantIntFromCleartextIfExists(mlir::Value cleartext) {
  auto constantOp = cleartext.getDefiningOp();
  if (constantOp == nullptr)
    return {};
  if (llvm::isa<arith::ConstantOp>(constantOp)) {
    auto constIntToMul = constantOp->getAttrOfType<mlir::IntegerAttr>("value");
    if (constIntToMul != nullptr)
      return constIntToMul;
  }
  return {};
}

/// Rewrite a TFHE multiplication with an integer operation as a
/// Zero operation if it's being multiplied with a constant 0, or as
/// a Negate operation if multiplied with a constant -1.
class MulCleartextLweCiphertextOpPattern
    : public mlir::OpRewritePattern<mlir::concretelang::TFHE::MulGLWEIntOp> {
public:
  MulCleartextLweCiphertextOpPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<mlir::concretelang::TFHE::MulGLWEIntOp>(
            context, ::mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::TFHE::MulGLWEIntOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cleartext = op.getOperand(1);
    auto constIntToMul = getConstantIntFromCleartextIfExists(cleartext);
    // Constant integer
    if (constIntToMul.has_value()) {
      auto toMul = constIntToMul.value().getInt();
      if (toMul == 0) {
        rewriter.replaceOpWithNewOp<mlir::concretelang::TFHE::ZeroGLWEOp>(
            op, op.getResult().getType());
        return mlir::success();
      }
      if (toMul == -1) {
        rewriter.replaceOpWithNewOp<mlir::concretelang::TFHE::NegGLWEOp>(
            op, op.getResult().getType(), op.getOperand(0));
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

/// Optimization pass that should choose more efficient ways of performing
/// crypto operations.
class TFHEOptimizationPass : public TFHEOptimizationBase<TFHEOptimizationPass> {
public:
  void runOnOperation() override {
    mlir::Operation *op = getOperation();

    mlir::RewritePatternSet patterns(op->getContext());
    patterns.add<MulCleartextLweCiphertextOpPattern>(op->getContext());

    if (mlir::applyPatternsAndFoldGreedily(op, std::move(patterns)).failed()) {
      this->signalPassFailure();
    }
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<>> createTFHEOptimizationPass() {
  return std::make_unique<TFHEOptimizationPass>();
}

} // namespace concretelang
} // namespace mlir

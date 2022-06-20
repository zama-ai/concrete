// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Arithmetic/IR/Arithmetic.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

#include <concretelang/Dialect/Concrete/IR/ConcreteOps.h>
#include <concretelang/Dialect/Concrete/Transforms/Optimization.h>
#include <concretelang/Support/Constants.h>

namespace mlir {
namespace concretelang {

namespace {

/// Get the constant integer that the cleartext was created from if it exists.
llvm::Optional<IntegerAttr>
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

/// Rewrite a `Concrete.mul_cleartext_lwe_ciphertext` operation as a
/// `Concrete.zero` operation if it's being multiplied with a constant 0, or as
/// a `Concrete.negate_lwe_ciphertext` if multiplied with a constant -1.
class MulCleartextLweCiphertextOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::MulCleartextLweCiphertextOp> {
public:
  MulCleartextLweCiphertextOpPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<
            mlir::concretelang::Concrete::MulCleartextLweCiphertextOp>(
            context, ::mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::MulCleartextLweCiphertextOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto cleartext = op.getOperand(1);
    auto constIntToMul = getConstantIntFromCleartextIfExists(cleartext);
    // Constant integer
    if (constIntToMul.hasValue()) {
      auto toMul = constIntToMul.getValue().getInt();
      if (toMul == 0) {
        rewriter.replaceOpWithNewOp<mlir::concretelang::Concrete::ZeroLWEOp>(
            op, op.getResult().getType());
        return mlir::success();
      }
      if (toMul == -1) {
        rewriter.replaceOpWithNewOp<
            mlir::concretelang::Concrete::NegateLweCiphertextOp>(
            op, op.getResult().getType(), op.getOperand(0));
        return mlir::success();
      }
    }
    return mlir::failure();
  }
};

/// Optimization pass that should choose more efficient ways of performing
/// crypto operations.
class ConcreteOptimizationPass
    : public ConcreteOptimizationBase<ConcreteOptimizationPass> {
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

std::unique_ptr<mlir::OperationPass<>> createConcreteOptimizationPass() {
  return std::make_unique<ConcreteOptimizationPass>();
}

} // namespace concretelang
} // namespace mlir

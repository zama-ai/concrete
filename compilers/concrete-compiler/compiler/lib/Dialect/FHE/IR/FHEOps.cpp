// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

namespace mlir {
namespace concretelang {
namespace FHE {

bool verifyEncryptedIntegerInputAndResultConsistency(
    mlir::Operation &op, FheIntegerInterface &input,
    FheIntegerInterface &result) {

  if (input.isSigned() != result.isSigned()) {
    op.emitOpError(
        "should have the signedness of encrypted inputs and result equal");
    return false;
  }

  if (input.getWidth() != result.getWidth()) {
    op.emitOpError(
        "should have the width of encrypted inputs and result equal");
    return false;
  }

  return true;
}

bool verifyEncryptedIntegerAndIntegerInputsConsistency(mlir::Operation &op,
                                                       FheIntegerInterface &a,
                                                       IntegerType &b) {
  if (a.getWidth() + 1 != b.getWidth()) {
    op.emitOpError("should have the width of plain input equal to width of "
                   "encrypted input + 1");
    return false;
  }

  return true;
}

bool verifyEncryptedIntegerInputsConsistency(mlir::Operation &op,
                                             FheIntegerInterface &a,
                                             FheIntegerInterface &b) {
  if (a.isSigned() != b.isSigned()) {
    op.emitOpError("should have the signedness of encrypted inputs equal");
    return false;
  }

  if (a.getWidth() != b.getWidth()) {
    op.emitOpError("should have the width of encrypted inputs equal");
    return false;
  }

  return true;
}

mlir::LogicalResult AddEintIntOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto b = this->getB().getType().cast<IntegerType>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return mlir::failure();
  }

  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         a, b)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult AddEintOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto b = this->getB().getType().dyn_cast<FheIntegerInterface>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }

  if (!verifyEncryptedIntegerInputsConsistency(*this->getOperation(), a, b)) {
    return ::mlir::failure();
  }

  return ::mlir::success();
}

mlir::LogicalResult SubIntEintOp::verify() {
  auto a = this->getA().getType().cast<IntegerType>();
  auto b = this->getB().getType().dyn_cast<FheIntegerInterface>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), b,
                                                       out)) {
    return mlir::failure();
  }

  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         b, a)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SubEintIntOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto b = this->getB().getType().cast<IntegerType>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return mlir::failure();
  }

  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         a, b)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SubEintOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto b = this->getB().getType().dyn_cast<FheIntegerInterface>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }

  if (!verifyEncryptedIntegerInputsConsistency(*this->getOperation(), a, b)) {
    return ::mlir::failure();
  }

  return ::mlir::success();
}

mlir::LogicalResult NegEintOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

mlir::LogicalResult MulEintIntOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto b = this->getB().getType().cast<IntegerType>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return mlir::failure();
  }

  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         a, b)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult MulEintOp::verify() {
  auto a = this->getRhs().getType().dyn_cast<FheIntegerInterface>();
  auto b = this->getLhs().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputsConsistency(*this->getOperation(), a, b)) {
    return ::mlir::failure();
  }

  return ::mlir::success();
}

mlir::LogicalResult MaxEintOp::verify() {
  auto xTy = this->getX().getType().dyn_cast<FheIntegerInterface>();
  auto yTy = this->getY().getType().dyn_cast<FheIntegerInterface>();
  auto outTy = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(),
                                                       xTy, outTy)) {
    return mlir::failure();
  }

  if (!verifyEncryptedIntegerInputsConsistency(*this->getOperation(), xTy,
                                               yTy)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ToSignedOp::verify() {
  auto input = this->getInput().getType().cast<EncryptedUnsignedIntegerType>();
  auto output = this->getResult().getType().cast<EncryptedSignedIntegerType>();

  if (input.getWidth() != output.getWidth()) {
    this->emitOpError(
        "should have the width of encrypted input and result equal");
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ToUnsignedOp::verify() {
  auto input = this->getInput().getType().cast<EncryptedSignedIntegerType>();
  auto output =
      this->getResult().getType().cast<EncryptedUnsignedIntegerType>();

  if (input.getWidth() != output.getWidth()) {
    this->emitOpError(
        "should have the width of encrypted input and result equal");
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult ToBoolOp::verify() {
  auto input = this->getInput().getType().cast<EncryptedUnsignedIntegerType>();

  if (input.getWidth() != 1 && input.getWidth() != 2) {
    this->emitOpError("should have 1 or 2 as the width of encrypted input to "
                      "cast to a boolean");
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult GenGateOp::verify() {
  auto truth_table = this->getTruthTable().getType().cast<TensorType>();

  mlir::SmallVector<int64_t, 1> expectedShape{4};
  if (!truth_table.hasStaticShape(expectedShape)) {
    this->emitOpError("truth table should be a tensor of 4 boolean values");
    return mlir::failure();
  }

  return mlir::success();
}

::mlir::LogicalResult ApplyLookupTableEintOp::verify() {
  auto ct = this->getA().getType().cast<FheIntegerInterface>();
  auto lut = this->getLut().getType().cast<TensorType>();

  // Check the shape of lut argument
  auto width = ct.getWidth();
  auto expectedSize = 1 << width;

  mlir::SmallVector<int64_t, 1> expectedShape{expectedSize};
  if (!lut.hasStaticShape(expectedShape)) {
    emitErrorBadLutSize(*this, "lut", "ct", expectedSize, width);
    return mlir::failure();
  }
  auto elmType = lut.getElementType();
  if (!elmType.isSignlessInteger() || elmType.getIntOrFloatBitWidth() > 64) {
    this->emitOpError() << "lut must have signless integer elements, with "
                           "precision not bigger than 64.";
    this->emitOpError() << "got : " << elmType.getIntOrFloatBitWidth();
    return mlir::failure();
  }
  return mlir::success();
}

mlir::LogicalResult RoundEintOp::verify() {
  auto input = this->getInput().getType().cast<FheIntegerInterface>();
  auto output = this->getResult().getType().cast<FheIntegerInterface>();

  if (input.getWidth() < output.getWidth()) {
    this->emitOpError(
        "should have the input width larger than the output width.");
    return mlir::failure();
  }

  if (input.isSigned() != output.isSigned()) {
    this->emitOpError(
        "should have the signedness of encrypted inputs and result equal");
    return mlir::failure();
  }

  return mlir::success();
}

OpFoldResult RoundEintOp::fold(FoldAdaptor operands) {
  auto input = this->getInput();
  auto inputTy = input.getType().dyn_cast_or_null<FheIntegerInterface>();
  auto outputTy = this->getResult().getType().cast<FheIntegerInterface>();
  if (inputTy.getWidth() == outputTy.getWidth()) {
    return input;
  }
  return nullptr;
}

/// Avoid addition with constant 0
OpFoldResult AddEintIntOp::fold(FoldAdaptor operands) {
  auto toAdd = operands.getB().dyn_cast_or_null<mlir::IntegerAttr>();
  if (toAdd != nullptr) {
    auto intToAdd = toAdd.getInt();
    if (intToAdd == 0) {
      return getOperand(0);
    }
  }
  return nullptr;
}

/// Avoid subtraction with constant 0
OpFoldResult SubEintIntOp::fold(FoldAdaptor operands) {
  auto toSub = operands.getB().dyn_cast_or_null<mlir::IntegerAttr>();
  if (toSub != nullptr) {
    auto intToSub = toSub.getInt();
    if (intToSub == 0) {
      return getOperand(0);
    }
  }
  return nullptr;
}

/// Avoid multiplication with constant 1
OpFoldResult MulEintIntOp::fold(FoldAdaptor operands) {
  auto toMul = operands.getB().dyn_cast_or_null<mlir::IntegerAttr>();
  if (toMul != nullptr) {
    auto intToMul = toMul.getInt();
    if (intToMul == 1) {
      return getOperand(0);
    }
  }
  return nullptr;
}

void MulEintIntOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {

  // Replace multiplication by clear zero cst to a trivial encrypted zero tensor
  class ZeroCstOpPattern : public mlir::OpRewritePattern<MulEintIntOp> {
  public:
    ZeroCstOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<MulEintIntOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(MulEintIntOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp = op.getB().getDefiningOp<arith::ConstantOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      auto val = cstOp->getAttrOfType<mlir::IntegerAttr>("value");
      if (val.getInt() != 0) {
        return mlir::failure();
      }
      rewriter.replaceOpWithNewOp<FHE::ZeroEintOp>(op,
                                                   op.getResult().getType());
      return mlir::success();
    }
  };

  // Replace multiplication by encrypted zero cst to a trivial encrypted zero
  // tensor
  class ZeroEncOpPattern : public mlir::OpRewritePattern<MulEintIntOp> {
  public:
    ZeroEncOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<MulEintIntOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(MulEintIntOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp = op.getA().getDefiningOp<FHE::ZeroEintOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      rewriter.replaceAllUsesWith(op, cstOp);
      rewriter.eraseOp(op);
      return mlir::success();
    }
  };
  patterns.add<ZeroCstOpPattern>(context);
  patterns.add<ZeroEncOpPattern>(context);
}

template <typename SignedConvOp>
void getSignedConvCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                           mlir::MLIRContext *context) {
  // Replace to_signed of zero to signed zero
  class ZeroOpPattern : public mlir::OpRewritePattern<SignedConvOp> {
  public:
    ZeroOpPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<SignedConvOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(SignedConvOp op,
                    mlir::PatternRewriter &rewriter) const override {
      auto cstOp = op.getInput().template getDefiningOp<FHE::ZeroEintOp>();
      if (cstOp == nullptr)
        return mlir::failure();
      rewriter.replaceOpWithNewOp<FHE::ZeroEintOp>(op,
                                                   op.getResult().getType());
      return mlir::success();
    }
  };
  patterns.add<ZeroOpPattern>(context);
}

void ToSignedOp::getCanonicalizationPatterns(mlir::RewritePatternSet &patterns,
                                             mlir::MLIRContext *context) {
  getSignedConvCanonicalizationPatterns<ToSignedOp>(patterns, context);
}

void ToUnsignedOp::getCanonicalizationPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *context) {
  getSignedConvCanonicalizationPatterns<ToUnsignedOp>(patterns, context);
}

} // namespace FHE
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/FHE/IR/FHEOps.cpp.inc"

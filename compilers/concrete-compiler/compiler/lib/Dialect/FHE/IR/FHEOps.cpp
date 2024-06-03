// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Support/CompilerEngine.h"

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
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
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
  auto b = this->getB().getType().dyn_cast<FheIntegerInterface>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), b,
                                                       out)) {
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult SubEintIntOp::verify() {
  auto a = this->getA().getType().dyn_cast<FheIntegerInterface>();
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
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
  auto out = this->getResult().getType().dyn_cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
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

mlir::LogicalResult ChangePartitionEintOp::verify() {
  auto input = this->getInput().getType().cast<FheIntegerInterface>();
  auto output = this->getResult().getType().cast<FheIntegerInterface>();

  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(),
                                                       input, output)) {
    return mlir::failure();
  }
  if (!verifyPartitionConsistency(this)) {
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

void ApplyLookupTableEintOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {

  class AfterTluPattern
      : public mlir::OpRewritePattern<ApplyLookupTableEintOp> {
  public:
    AfterTluPattern(mlir::MLIRContext *context)
        : mlir::OpRewritePattern<ApplyLookupTableEintOp>(context, 0) {}

    mlir::LogicalResult
    matchAndRewrite(ApplyLookupTableEintOp currentOperation,
                    mlir::PatternRewriter &rewriter) const override {

      CompilationOptions currentCompilationOptions =
          getCurrentCompilationOptions();

      if (!currentCompilationOptions.enableTluFusing) {
        return mlir::failure();
      }

      auto intermediateValue = currentOperation.getA();
      auto intermediateOperation =
          llvm::dyn_cast_or_null<ApplyLookupTableEintOp>(
              intermediateValue.getDefiningOp());

      if (!intermediateOperation) {
        return mlir::failure();
      }

      auto intermediateTableValue = intermediateOperation.getLut();
      auto intermediateTableOperation =
          llvm::dyn_cast_or_null<arith::ConstantOp>(
              intermediateTableValue.getDefiningOp());

      auto currentTableValue = currentOperation.getLut();
      auto currentTableOperation = llvm::dyn_cast_or_null<arith::ConstantOp>(
          currentTableValue.getDefiningOp());

      if (!intermediateTableOperation || !currentTableOperation) {
        return mlir::success();
      }

      auto intermediateTableContentAttr =
          (intermediateTableOperation.getValueAttr()
               .dyn_cast_or_null<mlir::DenseIntElementsAttr>());
      auto currentTableContentAttr =
          (currentTableOperation.getValueAttr()
               .dyn_cast_or_null<mlir::DenseIntElementsAttr>());

      if (!intermediateTableContentAttr || !currentTableContentAttr) {
        return mlir::failure();
      }

      auto intermediateTableContent =
          (intermediateTableContentAttr.getValues<int64_t>());
      auto currentTableContent = (currentTableContentAttr.getValues<int64_t>());

      auto inputValue = intermediateOperation.getA();
      auto inputType = inputValue.getType().dyn_cast<FheIntegerInterface>();
      auto inputBitWidth = inputType.getWidth();
      auto inputIsSigned = inputType.isSigned();

      auto intermediateType =
          (intermediateValue.getType().dyn_cast<FheIntegerInterface>());
      auto intermediateBitWidth = intermediateType.getWidth();
      auto intermediateIsSigned = intermediateType.isSigned();

      auto usersOfPreviousOperation = intermediateOperation->getUsers();
      auto numberOfUsersOfPreviousOperation = std::distance(
          usersOfPreviousOperation.begin(), usersOfPreviousOperation.end());

      if (numberOfUsersOfPreviousOperation > 1) {
        // This is a special case.
        //
        // Imagine you have this structure:
        // -----------------
        // x: uint6
        // y: uint3 = tlu[x]
        // z: uint3 = y + 1
        // a: uint3 = tlu[y]
        // b: uint3 = a + z
        // -----------------
        //
        // In this case, it's be better not to fuse `a = tlu[tlu[x]]`.
        //
        // The reason is that intermediate `y` is necessary for `z`,
        // so it has to be computed anyway.
        //
        // So to calculate `a`, there are 2 options:
        // - fused tlu on x
        // - regular tlu on y
        //
        // So for such cases, it's only better to fuse if the
        // bit width of `x` is smaller than the bit width of `y`.

        auto shouldFuse = inputBitWidth < intermediateBitWidth;
        if (!shouldFuse) {
          return mlir::failure();
        }
      }

      auto intermediateTableSize = 1 << inputBitWidth;
      auto currentTableSize = 1 << intermediateBitWidth;

      auto newTableContent = std::vector<int64_t>();
      newTableContent.reserve(intermediateTableSize);

      auto lookup = [&](ssize_t index) {
        if (index < 0) {
          index += intermediateTableSize;
        }
        auto resultOfFirstLookup = intermediateTableContent[index];

        // If the result of the first lookup is negative
        if (resultOfFirstLookup < 0) {
          // We first add the table size to preserve semantics
          // e.g., table[-1] == last element in the table == tableSize + (-1)
          // e.g., table[-2] == one element before that == tableSize + (-2)
          resultOfFirstLookup += currentTableSize;

          // If it's still negative
          if (resultOfFirstLookup < 0) {
            // e.g., imagine first table resulted in -100_000
            // (which can exist in tables...)
            // then we set it to the smallest possible value
            // of the input to the table

            // So if -100 is encountered on a signed 7-bit tlu
            // corresponding value will be calculated as if -64 is looked up

            // [0, 1, 2, 3, -4, -3, -2, -1]
            //              ^^ smallest value will always be in the middle

            resultOfFirstLookup = currentTableSize / 2;
          }
        } else if (resultOfFirstLookup >= currentTableSize) {
          // Another special case is the result of the first table
          // being bigger than the table itself

          // In this case we approach the value as the
          // biggest possible value of the input to the table

          if (!intermediateIsSigned) {

            // So if 100 is encountered on a unsigned 6-bit tlu
            // corresponding value will be calculated as if 63 is looked up

            // [0, 1, 2, 3, 4, 5, 6, 7]
            //                       ^ biggest value will always be in the end

            resultOfFirstLookup = currentTableSize - 1;

          } else {

            // So if 100 is encountered on a signed 7-bit tlu
            // corresponding value will be calculated as if 63 is looked up

            // [0, 1, 2, 3, -4, -3, -2, -1]
            //           ^ biggest value will always be in one before the middle

            resultOfFirstLookup = (currentTableSize / 2) - 1;
          }
        }
        auto resultOfSecondLookup = currentTableContent[resultOfFirstLookup];

        return resultOfSecondLookup;
      };

      if (!inputIsSigned) {
        // unsigned lookup table structure
        // [0, 1, 2, 3, 4, 5, 6, 7]
        // is the identity table

        // for the whole table
        for (ssize_t x = 0; x < intermediateTableSize; x++) {
          newTableContent.push_back(lookup(x));
        }
      } else {
        // signed lookup table structure
        // [0, 1, 2, 3, -4, -3, -2, -1]
        // is the identity table

        // for the positive part
        for (ssize_t x = 0; x < intermediateTableSize / 2; x++) {
          newTableContent.push_back(lookup(x));
        }
        // for the negative part
        for (ssize_t x = -(intermediateTableSize / 2); x < 0; x++) {
          newTableContent.push_back(lookup(x));
        }
      }

      auto newTable = rewriter.create<arith::ConstantOp>(
          currentOperation.getLoc(),
          DenseIntElementsAttr::get(intermediateTableValue.getType(),
                                    newTableContent));

      auto newOperation = rewriter.create<ApplyLookupTableEintOp>(
          currentOperation.getLoc(), currentOperation.getType(), inputValue,
          newTable);

      if (currentCompilationOptions.printTluFusing) {
        printTluFusing(currentOperation.getA(), currentOperation->getResult(0),
                       newOperation.getResult());
      }

      rewriter.replaceAllUsesWith(currentOperation, newOperation);
      return mlir::success();
    }
  };
  patterns.add<AfterTluPattern>(context);
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

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/FHE/Analysis/utils.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/Transforms/EncryptedMulToDoubleTLU/EncryptedMulToDoubleTLU.h>
#include <concretelang/Support/Constants.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>
#include <unordered_set>

using namespace mlir::concretelang::FHE;

namespace mlir {
namespace concretelang {
namespace {

class EncryptedMulOpPattern : public mlir::OpConversionPattern<FHE::MulEintOp> {
public:
  EncryptedMulOpPattern(mlir::MLIRContext *context)
      : mlir::OpConversionPattern<FHE::MulEintOp>(
            context, ::mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::MulEintOp op, FHE::MulEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    // Note: To understand the operator indexes propagation take a look at the
    // addMul function on ConcreteOptimizer.cpp

    auto inputType = adaptor.getRhs().getType();
    auto outputType = op->getResult(0).getType();

    auto bitWidth = inputType.cast<FHE::FheIntegerInterface>().getWidth();
    auto isSigned = inputType.cast<FHE::FheIntegerInterface>().isSigned();
    mlir::Type signedType =
        FHE::EncryptedSignedIntegerType::get(op->getContext(), bitWidth);

    // Get the operator indexes stored in FHE.mul operator to annotated FHE
    // nodes
    auto operatorIndexes =
        op->getAttrOfType<mlir::DenseI32ArrayAttr>("TFHE.OId");
    // Note:
    // -----
    //
    // The signedness of a value is only important:
    //     + when used as function input / output, because it changes the
    //       encoding/decoding used.
    //     + when used as tlu input, because it changes the encoding of the lut.
    //
    // Otherwise, for the leveled operations, the semantics are compatible. We
    // just have to please the verifier that usually requires the same
    // signedness for inputs and outputs.

    // s = a + b
    auto sum = rewriter.create<FHE::AddEintOp>(op->getLoc(), adaptor.getRhs(),
                                               adaptor.getLhs());
    if (operatorIndexes != nullptr)
      sum->setAttr("TFHE.OId", rewriter.getI32IntegerAttr(operatorIndexes[0]));
    // se = (s)^2/4
    // Depending on whether a,b,s are signed or not, we need a different lut to
    // compute (.)^2/4.
    mlir::SmallVector<uint64_t> rawSumLut;
    if (isSigned) {
      rawSumLut = generateSignedLut(bitWidth);
    } else {
      rawSumLut = generateUnsignedLut(bitWidth);
    }
    mlir::Value sumLut = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), mlir::DenseIntElementsAttr::get(
                          mlir::RankedTensorType::get(
                              rawSumLut.size(), rewriter.getIntegerType(64)),
                          rawSumLut));
    auto sumTluOutput = rewriter.create<FHE::ApplyLookupTableEintOp>(
        op->getLoc(), outputType, sum, sumLut);
    if (operatorIndexes != nullptr) {
      std::vector<int32_t> sumTluIndexes{operatorIndexes[1]};
      if (isSigned) {
        sumTluIndexes = {operatorIndexes[6], operatorIndexes[1]};
      }
      sumTluOutput->setAttr("TFHE.OId",
                            rewriter.getDenseI32ArrayAttr(sumTluIndexes));
    }

    // d = a - b
    auto diffOp = rewriter.create<FHE::SubEintOp>(
        op->getLoc(), adaptor.getRhs(), adaptor.getLhs());
    if (operatorIndexes != nullptr)
      diffOp->setAttr("TFHE.OId",
                      rewriter.getI32IntegerAttr(operatorIndexes[2]));
    mlir::Value diff = diffOp;

    // de = (d)^2/4
    // Here, the tlu must be performed with signed encoded lut, to properly
    // bootstrap negative values that may arise in the computation of d. If the
    // inputs are not signed, we cast the output to a signed encrypted integer.
    mlir::Value diffO;
    if (isSigned) {
      diffO = diff;
    } else {
      diff = rewriter.create<FHE::ToSignedOp>(op->getLoc(), signedType, diff);
    }
    mlir::SmallVector<uint64_t> rawDiffLut = generateSignedLut(bitWidth);
    mlir::Value diffLut = rewriter.create<mlir::arith::ConstantOp>(
        op->getLoc(), mlir::DenseIntElementsAttr::get(
                          mlir::RankedTensorType::get(
                              rawDiffLut.size(), rewriter.getIntegerType(64)),
                          rawDiffLut));
    auto diffTluOutput = rewriter.create<FHE::ApplyLookupTableEintOp>(
        op->getLoc(), outputType, diff, diffLut);
    if (operatorIndexes != nullptr) {
      std::vector<int32_t> diffTluIndexes{operatorIndexes[3],
                                          operatorIndexes[4]};
      diffTluOutput->setAttr("TFHE.OId",
                             rewriter.getDenseI32ArrayAttr(diffTluIndexes));
    }

    // o = se - de
    auto output = rewriter.create<FHE::SubEintOp>(op->getLoc(), outputType,
                                                  sumTluOutput, diffTluOutput);
    if (operatorIndexes != nullptr)
      output->setAttr("TFHE.OId",
                      rewriter.getI32IntegerAttr(operatorIndexes[5]));

    rewriter.replaceOp(op, {output});

    return mlir::success();
  }

private:
  static mlir::SmallVector<uint64_t> generateUnsignedLut(unsigned bitWidth) {
    mlir::SmallVector<uint64_t> rawLut;
    uint64_t lutLen = 1 << bitWidth;
    for (uint64_t i = 0; i < lutLen; ++i) {
      rawLut.push_back((i * i) / 4);
    }
    return rawLut;
  }

  static mlir::SmallVector<uint64_t> generateSignedLut(unsigned bitWidth) {
    mlir::SmallVector<uint64_t> rawLut;
    uint64_t lutLen = 1 << bitWidth;
    for (uint64_t i = 0; i < lutLen / 2; ++i) {
      rawLut.push_back((i * i) / 4);
    }
    for (uint64_t i = lutLen / 2; i > 0; --i) {
      rawLut.push_back((i * i) / 4);
    }
    return rawLut;
  }
};

} // namespace

/// This pass rewrites an `FHE::MulEintOp` into a set of ops of the `FHE`
/// dialects.
///
/// It relies on the observation that `x*y` can be turned into `((x+y)^2)/4 -
/// ((x-y)^2)/4`, which uses operations already available in the `FHE` dialect:
/// + `x+y` can be computed with the leveled operation `add_eint`
/// + `x-y` can be computed with the leveled operation `sub_eint`
/// + `(a^2)/4` can be computed with a table lookup `apply_table_lookup`
///
/// Gotchas:
/// --------
///
/// Since we use the leveled addition and subtraction, we have to increment the
/// bitwidth of the inputs to properly encode the carry of the computation. For
/// this reason the user must ensure that an extra bit is provided.
///
/// Assuming a precision of N, `(x+y)^2/4` may evaluate to a value that
/// overflows the container. It turns out that this is not important in this
/// particular case for the following reason. One can easily show that the
/// following property holds: (a+b) mod c = (a mod c + b mod c) mod c
///
/// In our case, all operations are reduced mod 2^N, and the result we want to
/// compute is:
/// ((x+y)^2/4 - (x-y)^2/4) mod 2^N
/// Which can be turned to:
/// ((x+y)^2/4 mod 2^N - (x-y)^2/4 mod 2^N) mod 2^N
///
/// It turns out this is exactly what we compute. (x+y)^2/4 and (x-y)^2/4 is
/// computed mod 2^N with the table lookup (because the output is N bit wide).
/// And the subtraction is also computed mod 2^N because it is also on N bits
/// wide eints.
class EncryptedMulToDoubleTLU
    : public EncryptedMulToDoubleTLUBase<EncryptedMulToDoubleTLU> {

public:
  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();

    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalDialect<FHE::FHEDialect>();
    target.addIllegalOp<FHE::MulEintOp>();

    mlir::RewritePatternSet patterns(funcOp->getContext());
    patterns.add<EncryptedMulOpPattern>(funcOp->getContext());
    if (mlir::applyPartialConversion(funcOp, target, std::move(patterns))
            .failed()) {
      funcOp->emitError("Failed to rewrite FHE mul_eint operation.");
      this->signalPassFailure();
    }
  }
};

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createEncryptedMulToDoubleTLUPass() {
  return std::make_unique<EncryptedMulToDoubleTLU>();
}

} // namespace concretelang
} // namespace mlir

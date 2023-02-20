// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/FHE/Analysis/utils.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/Transforms/EncryptedMulToDoubleTLU.h>
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

    auto inputType = adaptor.getA().getType();
    auto bitWidth = inputType.cast<FHE::FheIntegerInterface>().getWidth();
    auto isSigned = inputType.cast<FHE::FheIntegerInterface>().isSigned();
    mlir::Type signedType =
        FHE::EncryptedSignedIntegerType::get(op->getContext(), bitWidth);

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
    mlir::Value sum = rewriter.create<FHE::AddEintOp>(
        op->getLoc(), adaptor.getA(), adaptor.getB());

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
    mlir::Value sumTluOutput = rewriter.create<FHE::ApplyLookupTableEintOp>(
        op->getLoc(), inputType, sum, sumLut);

    // d = a - b
    mlir::Value diff = rewriter.create<FHE::SubEintOp>(
        op->getLoc(), adaptor.getA(), adaptor.getB());

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
    mlir::Value diffTluOutput = rewriter.create<FHE::ApplyLookupTableEintOp>(
        op->getLoc(), inputType, diff, diffLut);

    // o = se - de
    mlir::Value output = rewriter.create<FHE::SubEintOp>(
        op->getLoc(), inputType, sumTluOutput, diffTluOutput);

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
/// + Since we use the leveled addition and subtraction, we have to increment
/// the bitwidth of the inputs to properly
///   encode the carry of the computation. This change in bitwidth must then be
///   propagated to the whole graph, both upstream and downstream.
/// + This graph-wide update may reach existing `apply_lookup_table` operations,
/// which in turn will necessitate an
///   update of the size of the lookup table.
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

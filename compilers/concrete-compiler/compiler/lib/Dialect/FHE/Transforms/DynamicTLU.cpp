// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <concretelang/Dialect/FHE/Analysis/utils.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/Transforms/DynamicTLU/DynamicTLU.h>
#include <concretelang/Support/Constants.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
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

struct ApplyLookupTableEintOpPattern
    : public mlir::OpConversionPattern<FHE::ApplyLookupTableEintOp> {

  ApplyLookupTableEintOpPattern(mlir::MLIRContext *context)
      : mlir::OpConversionPattern<FHE::ApplyLookupTableEintOp>(
            context, ::mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::ApplyLookupTableEintOp op,
                  FHE::ApplyLookupTableEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    // When lowered to the TFHE dialect, the table will need to be properly
    // encoded by a function specific to the kind of table lookup executed. This
    // function expects the input lut to use 64 bit integers. For this reason,
    // every lut that use integers of smaller precision needs to be extended to
    // 64 bits first.

    bool outputIsSigned =
        op.getResult().getType().cast<FHE::FheIntegerInterface>().isSigned();
    auto inputLutType = op.getLut().getType();

    mlir::Value extendedLut;
    if (inputLutType.getElementType().getIntOrFloatBitWidth() == 64) {
      extendedLut = adaptor.getLut();
    } else {
      // This is implemented as a map since the `arith.extsi` is not
      // bufferizable :(
      mlir::Value init = rewriter.create<mlir::bufferization::AllocTensorOp>(
          op.getLoc(),
          mlir::RankedTensorType::get(inputLutType.getShape(),
                                      rewriter.getI64Type()),
          mlir::ValueRange{});

      extendedLut =
          rewriter
              .create<mlir::linalg::MapOp>(
                  op.getLoc(), mlir::ValueRange{adaptor.getLut()}, init,
                  [&](mlir::OpBuilder &builder, mlir::Location loc,
                      mlir::ValueRange args) {
                    mlir::Value extended;
                    if (outputIsSigned) {
                      extended = builder.create<mlir::arith::ExtSIOp>(
                          loc, builder.getI64Type(), args[0]);
                    } else {
                      extended = builder.create<mlir::arith::ExtUIOp>(
                          loc, builder.getI64Type(), args[0]);
                    }
                    builder.create<mlir::linalg::YieldOp>(
                        loc, mlir::ValueRange{extended});
                  })
              ->getResult(0);
    }

    auto newOp = rewriter.replaceOpWithNewOp<FHE::ApplyLookupTableEintOp>(
        op, op.getResult().getType(), op.getA(), extendedLut);

    // Propagating the Oid if any ...
    auto optimizerIdAttr = op->getAttr("TFHE.OId");
    if (optimizerIdAttr != nullptr)
      newOp->setAttr("TFHE.OId", optimizerIdAttr);

    return mlir::success();
  };
};

} // namespace

class DynamicTLU : public DynamicTLUBase<DynamicTLU> {

public:
  void runOnOperation() override {
    mlir::func::FuncOp funcOp = getOperation();
    mlir::ConversionTarget target(getContext());

    target.addLegalDialect<mlir::arith::ArithDialect>();
    target.addLegalOp<mlir::linalg::MapOp, mlir::linalg::YieldOp,
                      mlir::bufferization::AllocTensorOp>();
    target.addLegalDialect<FHE::FHEDialect>();
    target.addDynamicallyLegalOp<FHE::ApplyLookupTableEintOp>(
        [&](FHE::ApplyLookupTableEintOp op) {
          return op.getLut()
                     .getType()
                     .getElementType()
                     .getIntOrFloatBitWidth() == 64;
        });

    mlir::RewritePatternSet patterns(funcOp->getContext());

    patterns.add<ApplyLookupTableEintOpPattern>(funcOp->getContext());

    if (mlir::applyPartialConversion(funcOp, target, std::move(patterns))
            .failed()) {
      funcOp->emitError("Failed to extend dynamic luts.");
      this->signalPassFailure();
    }
  }
};

std::unique_ptr<::mlir::OperationPass<::mlir::func::FuncOp>>
createDynamicTLUPass() {
  return std::make_unique<DynamicTLU>();
}

} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Transforms/DialectConversion.h>

#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
#include <concretelang/Conversion/Utils/Legality.h>
#include <concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h>
#include <concretelang/Dialect/FHE/IR/FHEOps.h>
#include <concretelang/Dialect/FHE/IR/FHETypes.h>
#include <concretelang/Dialect/FHE/Transforms/BigInt/BigInt.h>
#include <concretelang/Support/Constants.h>

namespace mlir {
namespace concretelang {

/// Construct a table lookup to extract the carry bit
mlir::Value getTruthTableCarryExtract(mlir::PatternRewriter &rewriter,
                                      mlir::Location loc,
                                      unsigned int chunkSize,
                                      unsigned int chunkWidth) {
  auto tableSize = 1 << chunkSize;
  std::vector<llvm::APInt> values;
  values.reserve(tableSize);
  for (auto i = 0; i < tableSize; i++) {
    if (i < 1 << chunkWidth)
      values.push_back(llvm::APInt(1, 0, false));
    else
      values.push_back(llvm::APInt(1, 1, false));
  }
  auto truthTableAttr = mlir::DenseElementsAttr::get(
      mlir::RankedTensorType::get({tableSize}, rewriter.getIntegerType(64)),
      values);
  auto truthTable =
      rewriter.create<mlir::arith::ConstantOp>(loc, truthTableAttr);
  return truthTable.getResult();
}

namespace {

namespace typing {

/// Converts `FHE::ChunkedEncryptedInteger` into a tensor of
/// `FHE::EncryptedInteger`.
mlir::RankedTensorType
convertChunkedEint(mlir::MLIRContext *context,
                   FHE::EncryptedUnsignedIntegerType chunkedEint,
                   unsigned int chunkSize, unsigned int chunkWidth) {
  auto eint = FHE::EncryptedUnsignedIntegerType::get(context, chunkSize);
  auto bigIntWidth = chunkedEint.getWidth();
  assert(bigIntWidth % chunkWidth == 0 &&
         "chunkWidth must divide width of the big integer");
  auto numberOfChunks = bigIntWidth / chunkWidth;
  std::vector<int64_t> shape({numberOfChunks});
  return mlir::RankedTensorType::get(shape, eint);
}

/// The type converter used to transform `FHE` ops on chunked integers
class TypeConverter : public mlir::TypeConverter {

public:
  TypeConverter(unsigned int chunkSize, unsigned int chunkWidth) {
    addConversion([](mlir::Type type) { return type; });
    addConversion(
        [chunkSize, chunkWidth](FHE::EncryptedUnsignedIntegerType type) {
          if (type.getWidth() > chunkSize) {
            return (mlir::Type)convertChunkedEint(type.getContext(), type,
                                                  chunkSize, chunkWidth);
          } else {
            return (mlir::Type)type;
          }
        });
  }
};

} // namespace typing

class AddEintPattern
    : public mlir::OpConversionPattern<mlir::concretelang::FHE::AddEintOp> {
public:
  AddEintPattern(mlir::TypeConverter &converter, mlir::MLIRContext *context,
                 unsigned int chunkSize, unsigned int chunkWidth)
      : mlir::OpConversionPattern<mlir::concretelang::FHE::AddEintOp>(
            converter, context, ::mlir::concretelang::DEFAULT_PATTERN_BENEFIT),
        chunkSize(chunkSize), chunkWidth(chunkWidth) {}

  mlir::LogicalResult
  matchAndRewrite(FHE::AddEintOp op, FHE::AddEintOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto tensorType =
        adaptor.getA().getType().dyn_cast<mlir::RankedTensorType>();
    auto shape = tensorType.getShape();
    assert(shape.size() == 1 &&
           "chunked integer should be converted to flat tensors, but tensor "
           "have more than one dimension");
    auto eintChunkWidth = tensorType.getElementType()
                              .dyn_cast<FHE::EncryptedUnsignedIntegerType>()
                              .getWidth();
    assert(eintChunkWidth == chunkSize && "wrong tensor elements width");
    auto numberOfChunks = shape[0];

    mlir::Value carry =
        rewriter
            .create<FHE::ZeroEintOp>(op.getLoc(),
                                     FHE::EncryptedUnsignedIntegerType::get(
                                         rewriter.getContext(), chunkSize))
            .getResult();

    mlir::Value resultTensor =
        rewriter
            .create<FHE::ZeroTensorOp>(op.getLoc(), adaptor.getA().getType())
            .getResult();
    // used to shift the carry bit to the left
    mlir::Value twoPowerChunkSizeCst =
        rewriter
            .create<mlir::arith::ConstantIntOp>(op.getLoc(), 1 << chunkWidth,
                                                chunkSize + 1)
            .getResult();
    // Create the loop
    int64_t lb = 0, step = 1;
    auto forOp = rewriter.create<mlir::AffineForOp>(
        op.getLoc(), lb, numberOfChunks, step, resultTensor,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::Value iter,
            mlir::ValueRange args) {
          // add inputs with the previous carry (init to 0)
          mlir::Value leftEint = builder.create<mlir::tensor::ExtractOp>(
              loc, adaptor.getA(), iter);
          mlir::Value rightEint = builder.create<mlir::tensor::ExtractOp>(
              loc, adaptor.getB(), iter);
          mlir::Value result =
              builder.create<FHE::AddEintOp>(loc, leftEint, rightEint)
                  .getResult();
          mlir::Value resultWithCarry =
              builder.create<FHE::AddEintOp>(loc, result, carry).getResult();
          // compute the new carry: either 1 or 0
          carry =
              rewriter.create<mlir::concretelang::FHE::ApplyLookupTableEintOp>(
                  op.getLoc(),
                  FHE::EncryptedUnsignedIntegerType::get(rewriter.getContext(),
                                                         chunkSize),
                  resultWithCarry,
                  getTruthTableCarryExtract(rewriter, op.getLoc(), chunkSize,
                                            chunkWidth));
          // remove the carry bit from the result
          mlir::Value shiftedCarry =
              builder
                  .create<FHE::MulEintIntOp>(loc, carry, twoPowerChunkSizeCst)
                  .getResult();
          mlir::Value finalResult =
              builder.create<FHE::SubEintOp>(loc, resultWithCarry, shiftedCarry)
                  .getResult();
          // insert the result in the result tensor
          mlir::Value tensorResult = builder.create<mlir::tensor::InsertOp>(
              loc, finalResult, args[0], iter);
          builder.create<mlir::AffineYieldOp>(loc, tensorResult);
        });
    rewriter.replaceOp(op, forOp.getResult(0));
    return mlir::success();
  }

private:
  unsigned int chunkSize, chunkWidth;
};

/// Performs the transformation of big integer operations
class FHEBigIntTransformPass
    : public FHEBigIntTransformBase<FHEBigIntTransformPass> {
public:
  FHEBigIntTransformPass(unsigned int chunkSize, unsigned int chunkWidth)
      : chunkSize(chunkSize), chunkWidth(chunkWidth){};

  void runOnOperation() override {
    mlir::Operation *op = getOperation();
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());
    typing::TypeConverter converter(chunkSize, chunkWidth);

    // Legal ops created during pattern application
    target.addLegalOp<mlir::AffineForOp, mlir::AffineYieldOp,
                      mlir::arith::ConstantOp, mlir::arith::ConstantIndexOp,
                      FHE::ZeroEintOp, FHE::ZeroTensorOp, FHE::AddEintOp,
                      FHE::MulEintIntOp, FHE::SubEintOp,
                      FHE::ApplyLookupTableEintOp, mlir::tensor::ExtractOp,
                      mlir::tensor::InsertOp>();
    concretelang::addDynamicallyLegalTypeOp<FHE::AddEintOp>(target, converter);
    // Func ops are only legal with converted types
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return converter.isSignatureLegal(funcOp.getFunctionType()) &&
                 converter.isLegal(&funcOp.getBody());
        });
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);
    patterns.add<concretelang::TypeConvertingReinstantiationPattern<
        mlir::func::ReturnOp>>(patterns.getContext(), converter);
    concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(target,
                                                                  converter);

    patterns.add<AddEintPattern>(converter, &getContext(), chunkSize,
                                 chunkWidth);

    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }

private:
  unsigned int chunkSize, chunkWidth;
};

} // end anonymous namespace

std::unique_ptr<mlir::OperationPass<>>
createFHEBigIntTransformPass(unsigned int chunkSize, unsigned int chunkWidth) {
  assert(chunkSize >= chunkWidth + 1 &&
         "chunkSize must be greater than chunkWidth");
  return std::make_unique<FHEBigIntTransformPass>(chunkSize, chunkWidth);
}

} // namespace concretelang
} // namespace mlir

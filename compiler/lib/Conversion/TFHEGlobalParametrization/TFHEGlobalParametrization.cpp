// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Support/Constants.h"

namespace TFHE = mlir::concretelang::TFHE;

namespace {
struct TFHEGlobalParametrizationPass
    : public TFHEGlobalParametrizationBase<TFHEGlobalParametrizationPass> {
  TFHEGlobalParametrizationPass(mlir::concretelang::V0FHEContext &fheContext)
      : fheContext(fheContext){};
  void runOnOperation() final;
  mlir::concretelang::V0FHEContext &fheContext;
};
} // namespace

using mlir::concretelang::TFHE::GLWECipherTextType;

/// TFHEGlobalParametrizationTypeConverter is a TypeConverter that transform
/// `TFHE.glwe<{_,_,_}{p}>` to
/// `TFHE.glwe<{glweDimension,polynomialSize,bits}{p'}>`
class TFHEGlobalParametrizationTypeConverter : public mlir::TypeConverter {

public:
  TFHEGlobalParametrizationTypeConverter(
      mlir::concretelang::V0FHEContext &fheContext) {
    auto convertGLWECiphertextType =
        [](GLWECipherTextType type,
           mlir::concretelang::V0FHEContext &fheContext) {
          auto glweDimension = fheContext.parameter.getNBigGlweDimension();
          auto p = fheContext.constraint.p;
          if (type.getDimension() == (signed)glweDimension &&
              type.getP() == (signed)p) {
            return type;
          }
          return GLWECipherTextType::get(
              type.getContext(), glweDimension,
              1 /*for the v0, is always lwe ciphertext*/,
              64 /*for the v0 we handle only q=64*/, p);
        };
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      return convertGLWECiphertextType(type, fheContext);
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto glwe = type.getElementType().dyn_cast_or_null<GLWECipherTextType>();
      if (glwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(), convertGLWECiphertextType(glwe, fheContext));
      return r;
    });
  }
};

template <typename Op>
struct TFHEOpTypeConversionPattern : public mlir::OpRewritePattern<Op> {
  TFHEOpTypeConversionPattern(mlir::MLIRContext *context,
                              mlir::TypeConverter &typeConverter,
                              mlir::PatternBenefit benefit =
                                  mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<Op>(context, benefit),
        typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Type, 1> newResultTypes;
    if (typeConverter.convertTypes(op->getResultTypes(), newResultTypes)
            .failed()) {
      return mlir::failure();
    }
    rewriter.replaceOpWithNewOp<Op>(op, newResultTypes, op->getOperands());
    return mlir::success();
  };

private:
  mlir::TypeConverter &typeConverter;
};

struct KeySwitchGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::KeySwitchGLWEOp> {
  KeySwitchGLWEOpPattern(mlir::MLIRContext *context,
                         mlir::TypeConverter &converter,
                         mlir::concretelang::V0FHEContext &fheContext,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::KeySwitchGLWEOp>(context, benefit),
        converter(converter), fheContext(fheContext) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::KeySwitchGLWEOp ksOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Type, 1> newResultTypes;
    auto inputTy = ksOp.ciphertext().getType().cast<TFHE::GLWECipherTextType>();
    auto outputTy = rewriter.getType<TFHE::GLWECipherTextType>(
        fheContext.parameter.glweDimension, fheContext.parameter.nSmall, 64,
        inputTy.getP());
    rewriter.replaceOpWithNewOp<TFHE::KeySwitchGLWEOp>(
        ksOp, outputTy, ksOp.ciphertext(), fheContext.parameter.ksLevel,
        fheContext.parameter.ksLogBase);
    return mlir::success();
  };

private:
  mlir::TypeConverter &converter;
  mlir::concretelang::V0FHEContext &fheContext;
};

struct BootstrapGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::BootstrapGLWEOp> {
  BootstrapGLWEOpPattern(mlir::MLIRContext *context,
                         mlir::TypeConverter &converter,
                         mlir::concretelang::V0FHEContext &fheContext,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::BootstrapGLWEOp>(context, benefit),
        converter(converter), fheContext(fheContext) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::BootstrapGLWEOp bsOp,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<TFHE::BootstrapGLWEOp>(
        bsOp, converter.convertType(bsOp.result().getType()), bsOp.ciphertext(),
        bsOp.lookup_table(), fheContext.parameter.glweDimension,
        1 << fheContext.parameter.logPolynomialSize,
        fheContext.parameter.brLevel, fheContext.parameter.brLogBase);
    return mlir::success();
  };

private:
  mlir::TypeConverter &converter;
  mlir::concretelang::V0FHEContext &fheContext;
};

// This rewrite pattern transforms any instance of `TFHE.glwe_from_table` by
// parametrize GLWE return type and pad the table if the precision has been
// changed.
//
// Example:
//
// ```mlir
// %lut = arith.constant dense<[0, 1, 2, 3]> : tensor<4xi64>
// %0 = "TFHE.glwe_from_table" (%lut) : (tensor<4xi64>) ->
// !TFHE.glwe<{_,_,_}{2}>
// ```
//
// becomes:
//
// ```mlir
// %lut = arith.constant dense<[0, 1, 2, 3, 0, 1, 2, 3]> : tensor<8xi64>
// %0 = "TFHE.glwe_from_table" (%lut) : (tensor<8xi64>) ->
// !TFHE.glwe<{_,_,_}{3}>
// ```
struct GLWEFromTablePattern
    : public mlir::OpRewritePattern<TFHE::GLWEFromTableOp> {
  GLWEFromTablePattern(mlir::MLIRContext *context,
                       mlir::TypeConverter &converter,
                       mlir::concretelang::V0FHEContext &fheContext,
                       mlir::PatternBenefit benefit =
                           mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::GLWEFromTableOp>(context, benefit),
        converter(converter), fheContext(fheContext) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::GLWEFromTableOp glweOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto newTy = converter.convertType(glweOp.getType())
                     .cast<TFHE::GLWECipherTextType>();

    auto lutOp = glweOp.table();
    auto tableTy = lutOp.getType().cast<mlir::RankedTensorType>();

    auto expectedSize = 1 << newTy.getP();
    if (tableTy.getShape()[0] < expectedSize) {
      // Create a new padded lookup table
      auto constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
          lutOp.getDefiningOp());
      if (constantOp == nullptr) {
        glweOp.emitError() << "padding for non-constant operator is NYI";
        return mlir::failure();
      }
      mlir::DenseIntElementsAttr denseVals =
          constantOp->getAttrOfType<mlir::DenseIntElementsAttr>("value");
      if (denseVals == nullptr) {
        constantOp.emitError() << "value should be dense";
        return mlir::failure();
      }
      auto integerSize = 64;
      llvm::SmallVector<llvm::APInt> rawNewDenseVals(
          expectedSize, llvm::APInt(integerSize, 0));
      for (auto i = 0; i < expectedSize; i++) {
        rawNewDenseVals[i] = llvm::APInt(
            integerSize,
            denseVals.getFlatValue<llvm::APInt>(i % denseVals.size())
                .getZExtValue());
      }
      auto newDenseValsType = mlir::RankedTensorType::get(
          {expectedSize}, rewriter.getIntegerType(integerSize));
      auto newDenseVals =
          mlir::DenseIntElementsAttr::get(newDenseValsType, rawNewDenseVals);
      // Replace the lutOp by the new padded lookup table
      lutOp = rewriter.create<mlir::arith::ConstantOp>(constantOp.getLoc(),
                                                       newDenseVals);
    }
    rewriter.replaceOpWithNewOp<TFHE::GLWEFromTableOp>(glweOp, newTy, lutOp);
    return mlir::success();
  };

private:
  mlir::TypeConverter &converter;
  mlir::concretelang::V0FHEContext &fheContext;
};

template <typename Op>
void populateWithTFHEOpTypeConversionPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter) {
  patterns.add<TFHEOpTypeConversionPattern<Op>>(patterns.getContext(),
                                                typeConverter);
  target.addDynamicallyLegalOp<Op>(
      [&](Op op) { return typeConverter.isLegal(op->getResultTypes()); });
}

/// Populate the RewritePatternSet with all patterns that rewrite Concrete
/// operators to the corresponding function call to the `Concrete C API`.
void populateWithTFHEOpTypeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter,
    mlir::concretelang::V0Parameter &v0Parameter) {
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::ZeroGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::ZeroTensorGLWEOp>(patterns, target,
                                                  typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::AddGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::AddGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::SubIntGLWEOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::NegGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::MulGLWEIntOp>(patterns, target, typeConverter);
}

void TFHEGlobalParametrizationPass::runOnOperation() {
  auto op = this->getOperation();

  TFHEGlobalParametrizationTypeConverter converter(fheContext);

  // Parametrize
  {
    mlir::ConversionTarget target(getContext());
    mlir::OwningRewritePatternList patterns(&getContext());

    // function signature
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType()) &&
             converter.isLegal(&funcOp.getBody());
    });
    mlir::populateFuncOpTypeConversionPattern(patterns, converter);

    // Parametrize keyswitch bootstrap
    patterns.add<GLWEFromTablePattern>(&getContext(), converter, fheContext);
    target.addDynamicallyLegalOp<TFHE::GLWEFromTableOp>(
        [&](TFHE::GLWEFromTableOp op) {
          return converter.isLegal(op->getResultTypes());
        });
    target.addLegalOp<mlir::arith::ConstantOp>();
    patterns.add<KeySwitchGLWEOpPattern>(&getContext(), converter, fheContext);
    target.addDynamicallyLegalOp<TFHE::KeySwitchGLWEOp>(
        [&](TFHE::KeySwitchGLWEOp op) {
          return op.level() != (uint32_t)-1 && op.baseLog() != (uint32_t)-1;
        });
    patterns.add<BootstrapGLWEOpPattern>(&getContext(), converter, fheContext);
    target.addDynamicallyLegalOp<TFHE::BootstrapGLWEOp>(
        [&](TFHE::BootstrapGLWEOp op) {
          return converter.isLegal(op->getResultTypes());
        });

    // Add all patterns to convert TFHE types
    populateWithTFHEOpTypeConversionPatterns(patterns, target, converter,
                                             fheContext.parameter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::linalg::GenericOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::tensor::GenerateOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    patterns.add<RegionOpTypeConverterPattern<
        mlir::scf::ForOp, TFHEGlobalParametrizationTypeConverter>>(
        &getContext(), converter);
    mlir::concretelang::populateWithTensorTypeConverterPatterns(
        patterns, target, converter);

    // Conversion of RT Dialect Ops
    patterns.add<mlir::concretelang::GenericTypeConverterPattern<
        mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(),
                                                 converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowTaskOp>(target, converter);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTFHEGlobalParametrizationPass(
    mlir::concretelang::V0FHEContext &fheContext) {
  return std::make_unique<TFHEGlobalParametrizationPass>(fheContext);
}
} // namespace concretelang
} // namespace mlir

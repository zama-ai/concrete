// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Support/Constants.h"

namespace {
struct TFHEGlobalParametrizationPass
    : public TFHEGlobalParametrizationBase<
          TFHEGlobalParametrizationPass> {
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
        [](GLWECipherTextType type, mlir::concretelang::V0FHEContext &fheContext) {
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
  TFHEOpTypeConversionPattern(
      mlir::MLIRContext *context, mlir::TypeConverter &typeConverter,
      mlir::PatternBenefit benefit = mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
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

struct TFHEApplyLookupTableParametrizationPattern
    : public mlir::OpRewritePattern<mlir::concretelang::TFHE::ApplyLookupTable> {
  TFHEApplyLookupTableParametrizationPattern(
      mlir::MLIRContext *context, mlir::TypeConverter &typeConverter,
      mlir::concretelang::V0Parameter &v0Parameter,
      mlir::PatternBenefit benefit = mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::TFHE::ApplyLookupTable>(
            context, benefit),
        typeConverter(typeConverter), v0Parameter(v0Parameter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::TFHE::ApplyLookupTable op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Type, 1> newResultTypes;
    if (typeConverter.convertTypes(op->getResultTypes(), newResultTypes)
            .failed()) {
      return mlir::failure();
    }

    mlir::SmallVector<mlir::NamedAttribute, 6> newAttributes{
        mlir::NamedAttribute(
            rewriter.getIdentifier("glweDimension"),
            rewriter.getI32IntegerAttr(v0Parameter.glweDimension)),
        mlir::NamedAttribute(
            rewriter.getIdentifier("polynomialSize"),
            // TODO remove the shift when we have true polynomial size
            rewriter.getI32IntegerAttr(1 << v0Parameter.logPolynomialSize)),
        mlir::NamedAttribute(rewriter.getIdentifier("levelKS"),
                             rewriter.getI32IntegerAttr(v0Parameter.ksLevel)),
        mlir::NamedAttribute(rewriter.getIdentifier("baseLogKS"),
                             rewriter.getI32IntegerAttr(v0Parameter.ksLogBase)),
        mlir::NamedAttribute(rewriter.getIdentifier("levelBS"),
                             rewriter.getI32IntegerAttr(v0Parameter.brLevel)),
        mlir::NamedAttribute(rewriter.getIdentifier("baseLogBS"),
                             rewriter.getI32IntegerAttr(v0Parameter.brLogBase)),
        mlir::NamedAttribute(rewriter.getIdentifier("outputSizeKS"),
                             rewriter.getI32IntegerAttr(v0Parameter.nSmall)),
    };

    rewriter.replaceOpWithNewOp<mlir::concretelang::TFHE::ApplyLookupTable>(
        op, newResultTypes, op->getOperands(), newAttributes);

    return mlir::success();
  };

private:
  mlir::TypeConverter &typeConverter;
  mlir::concretelang::V0Parameter &v0Parameter;
};

struct TFHEApplyLookupTablePaddingPattern
    : public mlir::OpRewritePattern<mlir::concretelang::TFHE::ApplyLookupTable> {
  TFHEApplyLookupTablePaddingPattern(
      mlir::MLIRContext *context,
      mlir::PatternBenefit benefit = mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::concretelang::TFHE::ApplyLookupTable>(
            context, benefit),
        typeConverter(typeConverter), v0Parameter(v0Parameter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::TFHE::ApplyLookupTable op,
                  mlir::PatternRewriter &rewriter) const override {
    auto glweInType = op.getOperandTypes()[0]
                          .cast<mlir::concretelang::TFHE::GLWECipherTextType>();
    auto tabulatedLambdaType =
        op.l_cst().getType().cast<mlir::RankedTensorType>();
    auto glweOutType =
        op.getType().cast<mlir::concretelang::TFHE::GLWECipherTextType>();
    auto expectedSize = 1 << glweInType.getP();
    if (tabulatedLambdaType.getShape()[0] < expectedSize) {
      auto constantOp = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(
          op.l_cst().getDefiningOp());
      if (constantOp == nullptr) {
        op.emitError() << "padding for non-constant operator is NYI";
        return mlir::failure();
      }
      mlir::DenseIntElementsAttr denseVals =
          constantOp->getAttrOfType<mlir::DenseIntElementsAttr>("value");
      if (denseVals == nullptr) {
        op.emitError() << "value should be dense";
        return mlir::failure();
      }
      // Create the new constant dense op with padding
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
      auto newConstantOp = rewriter.create<mlir::arith::ConstantOp>(
          constantOp.getLoc(), newDenseVals);
      // Replace the apply_lookup_table with the new constant
      mlir::SmallVector<mlir::Type> newResultTypes{op.getType()};
      llvm::SmallVector<mlir::Value> newOperands{op.ct(), newConstantOp};
      llvm::ArrayRef<mlir::NamedAttribute> newAttrs = op->getAttrs();
      rewriter.replaceOpWithNewOp<mlir::concretelang::TFHE::ApplyLookupTable>(
          op, newResultTypes, newOperands, newAttrs);
      return mlir::success();
    }

    return mlir::success();
  };

private:
  mlir::TypeConverter &typeConverter;
  mlir::concretelang::V0Parameter &v0Parameter;
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

void populateWithTFHEApplyLookupTableParametrizationPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter,
    mlir::concretelang::V0Parameter &v0Parameter) {
  patterns.add<TFHEApplyLookupTableParametrizationPattern>(
      patterns.getContext(), typeConverter, v0Parameter);
  target.addDynamicallyLegalOp<mlir::concretelang::TFHE::ApplyLookupTable>(
      [&](mlir::concretelang::TFHE::ApplyLookupTable op) {
        if (op.glweDimension() != v0Parameter.glweDimension ||
            // TODO remove the shift when we have true polynomial size
            op.polynomialSize() != (1 << v0Parameter.logPolynomialSize) ||
            op.levelKS() != v0Parameter.ksLevel ||
            op.baseLogKS() != v0Parameter.ksLogBase ||
            op.levelBS() != v0Parameter.brLevel ||
            op.baseLogBS() != v0Parameter.brLogBase) {
          return false;
        }
        return typeConverter.isLegal(op->getResultTypes());
      });
}

void populateWithTFHEApplyLookupTablePaddingPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target) {
  patterns.add<TFHEApplyLookupTablePaddingPattern>(patterns.getContext());
  target.addLegalOp<mlir::arith::ConstantOp>();
  target.addDynamicallyLegalOp<mlir::concretelang::TFHE::ApplyLookupTable>(
      [&](mlir::concretelang::TFHE::ApplyLookupTable op) {
        auto glweInType =
            op.getOperandTypes()[0]
                .cast<mlir::concretelang::TFHE::GLWECipherTextType>();
        auto tabulatedLambdaType =
            op.getOperandTypes()[1].cast<mlir::RankedTensorType>();
        auto glweOutType =
            op.getType().cast<mlir::concretelang::TFHE::GLWECipherTextType>();

        return tabulatedLambdaType.getShape()[0] == 1 << glweInType.getP();
      });
}

/// Populate the RewritePatternSet with all patterns that rewrite Concrete
/// operators to the corresponding function call to the `Concrete C API`.
void populateWithTFHEOpTypeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter,
    mlir::concretelang::V0Parameter &v0Parameter) {
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::ZeroGLWEOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::AddGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::AddGLWEOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::SubIntGLWEOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::NegGLWEOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::MulGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEApplyLookupTableParametrizationPattern(
      patterns, target, typeConverter, v0Parameter);
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
    mlir::concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                            converter);

    // Conversion of RT Dialect Ops
    patterns.add<mlir::concretelang::GenericTypeConverterPattern<
        mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(), converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowTaskOp>(target, converter);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }

  // Pad lookup table
  {
    mlir::ConversionTarget target(getContext());
    mlir::OwningRewritePatternList patterns(&getContext());

    populateWithTFHEApplyLookupTablePaddingPattern(patterns, target);

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

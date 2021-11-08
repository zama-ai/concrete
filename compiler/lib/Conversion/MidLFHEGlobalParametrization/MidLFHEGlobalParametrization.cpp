#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Conversion/Utils/LinalgGenericTypeConverterPattern.h"
#include "zamalang/Conversion/Utils/TensorOpTypeConversion.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace {
struct MidLFHEGlobalParametrizationPass
    : public MidLFHEGlobalParametrizationBase<
          MidLFHEGlobalParametrizationPass> {
  MidLFHEGlobalParametrizationPass(mlir::zamalang::V0FHEContext &fheContext)
      : fheContext(fheContext){};
  void runOnOperation() final;
  mlir::zamalang::V0FHEContext &fheContext;
};
} // namespace

using mlir::zamalang::MidLFHE::GLWECipherTextType;

/// MidLFHEGlobalParametrizationTypeConverter is a TypeConverter that transform
/// `MidLFHE.gwle<{_,_,_}{p}>` to
/// `MidLFHE.gwle<{glweSize,polynomialSize,bits}{p'}>`
class MidLFHEGlobalParametrizationTypeConverter : public mlir::TypeConverter {

public:
  MidLFHEGlobalParametrizationTypeConverter(
      mlir::zamalang::V0FHEContext &fheContext) {
    auto convertGLWECiphertextType =
        [](GLWECipherTextType type, mlir::zamalang::V0FHEContext &fheContext) {
          auto glweSize = fheContext.parameter.getNBigGlweSize();
          auto p = fheContext.constraint.p;
          if (type.getDimension() == glweSize && type.getP() == p) {
            return type;
          }
          return GLWECipherTextType::get(
              type.getContext(), glweSize,
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
struct MidLFHEOpTypeConversionPattern : public mlir::OpRewritePattern<Op> {
  MidLFHEOpTypeConversionPattern(mlir::MLIRContext *context,
                                 mlir::TypeConverter &typeConverter,
                                 mlir::PatternBenefit benefit = 1)
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

struct MidLFHEApplyLookupTableParametrizationPattern
    : public mlir::OpRewritePattern<mlir::zamalang::MidLFHE::ApplyLookupTable> {
  MidLFHEApplyLookupTableParametrizationPattern(
      mlir::MLIRContext *context, mlir::TypeConverter &typeConverter,
      mlir::zamalang::V0Parameter &v0Parameter,
      mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::MidLFHE::ApplyLookupTable>(
            context, benefit),
        typeConverter(typeConverter), v0Parameter(v0Parameter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::MidLFHE::ApplyLookupTable op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::SmallVector<mlir::Type, 1> newResultTypes;
    if (typeConverter.convertTypes(op->getResultTypes(), newResultTypes)
            .failed()) {
      return mlir::failure();
    }

    mlir::SmallVector<mlir::NamedAttribute, 6> newAttributes{
        mlir::NamedAttribute(rewriter.getIdentifier("k"),
                             rewriter.getI32IntegerAttr(v0Parameter.k)),
        mlir::NamedAttribute(
            rewriter.getIdentifier("polynomialSize"),
            // TODO remove the shift when we have true polynomial size
            rewriter.getI32IntegerAttr(1 << v0Parameter.polynomialSize)),
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

    rewriter.replaceOpWithNewOp<mlir::zamalang::MidLFHE::ApplyLookupTable>(
        op, newResultTypes, op->getOperands(), newAttributes);

    return mlir::success();
  };

private:
  mlir::TypeConverter &typeConverter;
  mlir::zamalang::V0Parameter &v0Parameter;
};

struct MidLFHEApplyLookupTablePaddingPattern
    : public mlir::OpRewritePattern<mlir::zamalang::MidLFHE::ApplyLookupTable> {
  MidLFHEApplyLookupTablePaddingPattern(mlir::MLIRContext *context,
                                        mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::zamalang::MidLFHE::ApplyLookupTable>(
            context, benefit),
        typeConverter(typeConverter), v0Parameter(v0Parameter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::MidLFHE::ApplyLookupTable op,
                  mlir::PatternRewriter &rewriter) const override {
    auto glweInType = op.getOperandTypes()[0]
                          .cast<mlir::zamalang::MidLFHE::GLWECipherTextType>();
    auto tabulatedLambdaType =
        op.l_cst().getType().cast<mlir::RankedTensorType>();
    auto glweOutType =
        op.getType().cast<mlir::zamalang::MidLFHE::GLWECipherTextType>();
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
      rewriter.replaceOpWithNewOp<mlir::zamalang::MidLFHE::ApplyLookupTable>(
          op, newResultTypes, newOperands, newAttrs);
      return mlir::success();
    }

    return mlir::success();
  };

private:
  mlir::TypeConverter &typeConverter;
  mlir::zamalang::V0Parameter &v0Parameter;
};

template <typename Op>
void populateWithMidLFHEOpTypeConversionPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter) {
  patterns.add<MidLFHEOpTypeConversionPattern<Op>>(patterns.getContext(),
                                                   typeConverter);
  target.addDynamicallyLegalOp<Op>(
      [&](Op op) { return typeConverter.isLegal(op->getResultTypes()); });
}

void populateWithMidLFHEApplyLookupTableParametrizationPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter,
    mlir::zamalang::V0Parameter &v0Parameter) {
  patterns.add<MidLFHEApplyLookupTableParametrizationPattern>(
      patterns.getContext(), typeConverter, v0Parameter);
  target.addDynamicallyLegalOp<mlir::zamalang::MidLFHE::ApplyLookupTable>(
      [&](mlir::zamalang::MidLFHE::ApplyLookupTable op) {
        if (op.k() != v0Parameter.k ||
            // TODO remove the shift when we have true polynomial size
            op.polynomialSize() != (1 << v0Parameter.polynomialSize) ||
            op.levelKS() != v0Parameter.ksLevel ||
            op.baseLogKS() != v0Parameter.ksLogBase ||
            op.levelBS() != v0Parameter.brLevel ||
            op.baseLogBS() != v0Parameter.brLogBase) {
          return false;
        }
        return typeConverter.isLegal(op->getResultTypes());
      });
}

void populateWithMidLFHEApplyLookupTablePaddingPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target) {
  patterns.add<MidLFHEApplyLookupTablePaddingPattern>(patterns.getContext());
  target.addLegalOp<mlir::arith::ConstantOp>();
  target.addDynamicallyLegalOp<mlir::zamalang::MidLFHE::ApplyLookupTable>(
      [&](mlir::zamalang::MidLFHE::ApplyLookupTable op) {
        auto glweInType =
            op.getOperandTypes()[0]
                .cast<mlir::zamalang::MidLFHE::GLWECipherTextType>();
        auto tabulatedLambdaType =
            op.getOperandTypes()[1].cast<mlir::RankedTensorType>();
        auto glweOutType =
            op.getType().cast<mlir::zamalang::MidLFHE::GLWECipherTextType>();

        return tabulatedLambdaType.getShape()[0] == 1 << glweInType.getP();
      });
}

/// Populate the RewritePatternSet with all patterns that rewrite LowLFHE
/// operators to the corresponding function call to the `Concrete C API`.
void populateWithMidLFHEOpTypeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter,
    mlir::zamalang::V0Parameter &v0Parameter) {
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::ZeroGLWEOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::AddGLWEIntOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::AddGLWEOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::SubIntGLWEOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::NegGLWEOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::MulGLWEIntOp>(patterns, target, typeConverter);
  populateWithMidLFHEApplyLookupTableParametrizationPattern(
      patterns, target, typeConverter, v0Parameter);
}

void MidLFHEGlobalParametrizationPass::runOnOperation() {
  auto op = this->getOperation();

  MidLFHEGlobalParametrizationTypeConverter converter(fheContext);

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

    // Add all patterns to convert MidLFHE types
    populateWithMidLFHEOpTypeConversionPatterns(patterns, target, converter,
                                                fheContext.parameter);
    patterns.add<LinalgGenericTypeConverterPattern<
        MidLFHEGlobalParametrizationTypeConverter>>(&getContext(), converter);
    mlir::zamalang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                            converter);

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

    populateWithMidLFHEApplyLookupTablePaddingPattern(patterns, target);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace zamalang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertMidLFHEGlobalParametrizationPass(
    mlir::zamalang::V0FHEContext &fheContext) {
  return std::make_unique<MidLFHEGlobalParametrizationPass>(fheContext);
}
} // namespace zamalang
} // namespace mlir

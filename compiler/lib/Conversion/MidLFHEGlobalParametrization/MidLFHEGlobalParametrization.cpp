#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/Passes.h"
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
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      auto glweSize = fheContext.parameter.getNBigGlweSize();
      auto p = fheContext.constraint.p;
      if (type.getDimension() == glweSize && type.getP() == p) {
        return type;
      }
      return GLWECipherTextType::get(type.getContext(), glweSize,
                                     1 /*for the v0, is always lwe ciphertext*/,
                                     64 /*for the v0 we handle only q=64*/, p);
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
            rewriter.getI32IntegerAttr(v0Parameter.polynomialSize)),
        mlir::NamedAttribute(rewriter.getIdentifier("levelKS"),
                             rewriter.getI32IntegerAttr(v0Parameter.ksLevel)),
        mlir::NamedAttribute(rewriter.getIdentifier("baseLogKS"),
                             rewriter.getI32IntegerAttr(v0Parameter.ksLogBase)),
        mlir::NamedAttribute(rewriter.getIdentifier("levelBS"),
                             rewriter.getI32IntegerAttr(v0Parameter.brLevel)),
        mlir::NamedAttribute(rewriter.getIdentifier("baseLogBS"),
                             rewriter.getI32IntegerAttr(v0Parameter.brLogBase)),
    };

    rewriter.replaceOpWithNewOp<mlir::zamalang::MidLFHE::ApplyLookupTable>(
        op, newResultTypes, op->getOperands(), newAttributes);

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
            op.polynomialSize() != v0Parameter.polynomialSize ||
            op.levelKS() != v0Parameter.ksLevel ||
            op.baseLogKS() != v0Parameter.ksLogBase ||
            op.levelBS() != v0Parameter.brLevel ||
            op.baseLogBS() != v0Parameter.brLogBase) {
          return false;
        }
        return typeConverter.isLegal(op->getResultTypes());
      });
}

/// Populate the RewritePatternSet with all patterns that rewrite LowLFHE
/// operators to the corresponding function call to the `Concrete C API`.
void populateWithMidLFHEOpTypeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter,
    mlir::zamalang::V0Parameter &v0Parameter) {
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::AddGLWEIntOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::AddGLWEOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::SubIntGLWEOp>(patterns, target, typeConverter);
  populateWithMidLFHEOpTypeConversionPattern<
      mlir::zamalang::MidLFHE::MulGLWEIntOp>(patterns, target, typeConverter);
  populateWithMidLFHEApplyLookupTableParametrizationPattern(
      patterns, target, typeConverter, v0Parameter);
}

void MidLFHEGlobalParametrizationPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  MidLFHEGlobalParametrizationTypeConverter converter(fheContext);

  // Make sure that no ops from `MidLFHE` remain after the lowering
  target.addIllegalDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();

  // Make sure func has legal signature
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
    return converter.isSignatureLegal(funcOp.getType()) &&
           converter.isLegal(&funcOp.getBody());
  });
  // Add all patterns required to lower all ops from `MidLFHE` to
  // `LowLFHE`
  mlir::OwningRewritePatternList patterns(&getContext());
  populateWithMidLFHEOpTypeConversionPatterns(patterns, target, converter,
                                              fheContext.parameter);
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
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

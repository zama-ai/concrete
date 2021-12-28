// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "concretelang/Dialect/LowLFHE/IR/LowLFHEOps.h"
#include "concretelang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Support/Constants.h"

/// LowLFHEUnparametrizeTypeConverter is a type converter that unparametrize
/// LowLFHE types
class LowLFHEUnparametrizeTypeConverter : public mlir::TypeConverter {

public:
  static mlir::Type unparematrizeLowLFHEType(mlir::Type type) {
    if (type.isa<mlir::concretelang::LowLFHE::PlaintextType>()) {
      return mlir::IntegerType::get(type.getContext(), 64);
    }
    if (type.isa<mlir::concretelang::LowLFHE::CleartextType>()) {
      return mlir::IntegerType::get(type.getContext(), 64);
    }
    if (type.isa<mlir::concretelang::LowLFHE::LweCiphertextType>()) {
      return mlir::concretelang::LowLFHE::LweCiphertextType::get(type.getContext(),
                                                             -1, -1);
    }
    auto tensorType = type.dyn_cast_or_null<mlir::RankedTensorType>();
    if (tensorType != nullptr) {
      auto eltTy0 = tensorType.getElementType();
      auto eltTy1 = unparematrizeLowLFHEType(eltTy0);
      if (eltTy0 == eltTy1) {
        return type;
      }
      return mlir::RankedTensorType::get(tensorType.getShape(), eltTy1);
    }
    return type;
  }

  LowLFHEUnparametrizeTypeConverter() {
    addConversion(
        [](mlir::Type type) { return unparematrizeLowLFHEType(type); });
  }
};

/// Replace `%1 = unrealized_conversion_cast %0 : t0 to t1` to `%0` where t0 or
/// t1 are a LowLFHE type.
struct LowLFHEUnrealizedCastReplacementPattern
    : public mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp> {
  LowLFHEUnrealizedCastReplacementPattern(
      mlir::MLIRContext *context,
      mlir::PatternBenefit benefit = mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<mlir::UnrealizedConversionCastOp>(context,
                                                                 benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::UnrealizedConversionCastOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (mlir::isa<mlir::concretelang::LowLFHE::LowLFHEDialect>(
            op.getOperandTypes()[0].getDialect()) ||
        mlir::isa<mlir::concretelang::LowLFHE::LowLFHEDialect>(
            op.getType(0).getDialect())) {
      rewriter.replaceOp(op, op.getOperands());
      return mlir::success();
    }
    return mlir::failure();
  };
};

/// LowLFHEUnparametrizePass remove all parameters of LowLFHE types and remove
/// the unrealized_conversion_cast operation that operates on parametrized
/// LowLFHE types.
struct LowLFHEUnparametrizePass
    : public LowLFHEUnparametrizeBase<LowLFHEUnparametrizePass> {
  void runOnOperation() final;
};

void LowLFHEUnparametrizePass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  mlir::OwningRewritePatternList patterns(&getContext());

  LowLFHEUnparametrizeTypeConverter converter;

  // Conversion of linalg.generic operation
  target
      .addDynamicallyLegalOp<mlir::linalg::GenericOp, mlir::tensor::GenerateOp>(
          [&](mlir::Operation *op) {
            return (
                converter.isLegal(op->getOperandTypes()) &&
                converter.isLegal(op->getResultTypes()) &&
                converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
          });
  patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                            LowLFHEUnparametrizeTypeConverter>>(
      &getContext(), converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                            LowLFHEUnparametrizeTypeConverter>>(
      &getContext(), converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                            LowLFHEUnparametrizeTypeConverter>>(
      &getContext(), converter);

  // Conversion of function signature and arguments
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
    return converter.isSignatureLegal(funcOp.getType()) &&
           converter.isLegal(&funcOp.getBody());
  });
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

  // Replacement of unrealized_conversion_cast
  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::UnrealizedConversionCastOp>(
      target, converter);
  patterns.add<LowLFHEUnrealizedCastReplacementPattern>(patterns.getContext());

  // Conversion of tensor operators
  mlir::concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                          converter);

  // Conversion of CallOp
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<mlir::CallOp>>(
      patterns.getContext(), converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::CallOp>(target, converter);

  // Conversion of RT Dialect Ops
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<
      mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(), converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::concretelang::RT::DataflowTaskOp>(
      target, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertLowLFHEUnparametrizePass() {
  return std::make_unique<LowLFHEUnparametrizePass>();
}
} // namespace concretelang
} // namespace mlir

#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/MidLFHEToLowLFHE/Patterns.h"
#include "zamalang/Conversion/Passes.h"
#include "zamalang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "zamalang/Conversion/Utils/TensorOpTypeConversion.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace {
struct MidLFHEToLowLFHEPass
    : public MidLFHEToLowLFHEBase<MidLFHEToLowLFHEPass> {
  void runOnOperation() final;
};
} // namespace

using mlir::zamalang::LowLFHE::LweCiphertextType;
using mlir::zamalang::MidLFHE::GLWECipherTextType;

/// MidLFHEToLowLFHETypeConverter is a TypeConverter that transform
/// `MidLFHE.glwe<{_,_,_}{p}>` to LowLFHE.lwe_ciphertext
class MidLFHEToLowLFHETypeConverter : public mlir::TypeConverter {

public:
  MidLFHEToLowLFHETypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      return mlir::zamalang::convertTypeToLWE(type.getContext(), type);
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto glwe = type.getElementType().dyn_cast_or_null<GLWECipherTextType>();
      if (glwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(),
          mlir::zamalang::convertTypeToLWE(glwe.getContext(), glwe));
      return r;
    });
  }
};

void MidLFHEToLowLFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  MidLFHEToLowLFHETypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::zamalang::LowLFHE::LowLFHEDialect>();

  // Make sure that no ops from `MidLFHE` remain after the lowering
  target.addIllegalDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();

  // Make sure that no ops `linalg.generic` that have illegal types
  target
      .addDynamicallyLegalOp<mlir::linalg::GenericOp, mlir::tensor::GenerateOp>(
          [&](mlir::Operation *op) {
            return (
                converter.isLegal(op->getOperandTypes()) &&
                converter.isLegal(op->getResultTypes()) &&
                converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
          });

  // Make sure that func has legal signature
  target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
    return converter.isSignatureLegal(funcOp.getType()) &&
           converter.isLegal(&funcOp.getBody());
  });
  // Add all patterns required to lower all ops from `MidLFHE` to
  // `LowLFHE`
  mlir::OwningRewritePatternList patterns(&getContext());

  populateWithGeneratedMidLFHEToLowLFHE(patterns);
  patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                            MidLFHEToLowLFHETypeConverter>>(
      &getContext(), converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                            MidLFHEToLowLFHETypeConverter>>(
      &getContext(), converter);
  mlir::zamalang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                          converter);
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace zamalang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertMidLFHEToLowLFHEPass() {
  return std::make_unique<MidLFHEToLowLFHEPass>();
}
} // namespace zamalang
} // namespace mlir

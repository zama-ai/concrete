// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/HLFHEToMidLFHE/Patterns.h"
#include "zamalang/Conversion/Passes.h"
#include "zamalang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "zamalang/Conversion/Utils/TensorOpTypeConversion.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"
#include "zamalang/Dialect/RT/IR/RTOps.h"

namespace {
struct HLFHEToMidLFHEPass : public HLFHEToMidLFHEBase<HLFHEToMidLFHEPass> {
  void runOnOperation() final;
};
} // namespace

using mlir::zamalang::HLFHE::EncryptedIntegerType;
using mlir::zamalang::MidLFHE::GLWECipherTextType;

/// HLFHEToMidLFHETypeConverter is a TypeConverter that transform
/// `HLFHE.eint<p>` to `MidLFHE.glwe<{_,_,_}{p}>`
class HLFHEToMidLFHETypeConverter : public mlir::TypeConverter {

public:
  HLFHEToMidLFHETypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([](EncryptedIntegerType type) {
      return mlir::zamalang::convertTypeEncryptedIntegerToGLWE(
          type.getContext(), type);
    });
    addConversion([](mlir::RankedTensorType type) {
      auto eint =
          type.getElementType().dyn_cast_or_null<EncryptedIntegerType>();
      if (eint == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(), mlir::zamalang::convertTypeEncryptedIntegerToGLWE(
                               eint.getContext(), eint));
      return r;
    });
  }
};

void HLFHEToMidLFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  HLFHEToMidLFHETypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::zamalang::MidLFHE::MidLFHEDialect>();

  // Make sure that no ops from `HLFHE` remain after the lowering
  target.addIllegalDialect<mlir::zamalang::HLFHE::HLFHEDialect>();

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
  // Add all patterns required to lower all ops from `HLFHE` to
  // `MidLFHE`
  mlir::OwningRewritePatternList patterns(&getContext());

  populateWithGeneratedHLFHEToMidLFHE(patterns);
  patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                            HLFHEToMidLFHETypeConverter>>(
      &getContext(), converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                            HLFHEToMidLFHETypeConverter>>(
      &getContext(), converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                            HLFHEToMidLFHETypeConverter>>(
      &getContext(), converter);

  mlir::zamalang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                          converter);
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

  // Conversion of RT Dialect Ops
  patterns.add<mlir::zamalang::GenericTypeConverterPattern<
      mlir::zamalang::RT::DataflowTaskOp>>(patterns.getContext(), converter);
  mlir::zamalang::addDynamicallyLegalTypeOp<mlir::zamalang::RT::DataflowTaskOp>(
      target, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace zamalang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertHLFHEToMidLFHEPass() {
  return std::make_unique<HLFHEToMidLFHEPass>();
}
} // namespace zamalang
} // namespace mlir

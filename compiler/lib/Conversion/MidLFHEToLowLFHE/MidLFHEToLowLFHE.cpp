// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/MidLFHEToLowLFHE/Patterns.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "concretelang/Dialect/LowLFHE/IR/LowLFHETypes.h"
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "concretelang/Dialect/MidLFHE/IR/MidLFHETypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"

namespace {
struct MidLFHEToLowLFHEPass
    : public MidLFHEToLowLFHEBase<MidLFHEToLowLFHEPass> {
  void runOnOperation() final;
};
} // namespace

using mlir::concretelang::LowLFHE::LweCiphertextType;
using mlir::concretelang::MidLFHE::GLWECipherTextType;

/// MidLFHEToLowLFHETypeConverter is a TypeConverter that transform
/// `MidLFHE.glwe<{_,_,_}{p}>` to LowLFHE.lwe_ciphertext
class MidLFHEToLowLFHETypeConverter : public mlir::TypeConverter {

public:
  MidLFHEToLowLFHETypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      return mlir::concretelang::convertTypeToLWE(type.getContext(), type);
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto glwe = type.getElementType().dyn_cast_or_null<GLWECipherTextType>();
      if (glwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(),
          mlir::concretelang::convertTypeToLWE(glwe.getContext(), glwe));
      return r;
    });
  }
};

void MidLFHEToLowLFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  MidLFHEToLowLFHETypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::concretelang::LowLFHE::LowLFHEDialect>();

  // Make sure that no ops from `MidLFHE` remain after the lowering
  target.addIllegalDialect<mlir::concretelang::MidLFHE::MidLFHEDialect>();

  // Make sure that no ops `linalg.generic` that have illegal types
  target.addDynamicallyLegalOp<mlir::linalg::GenericOp,
                               mlir::tensor::GenerateOp, mlir::scf::ForOp>(
      [&](mlir::Operation *op) {
        return (converter.isLegal(op->getOperandTypes()) &&
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
  patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                            MidLFHEToLowLFHETypeConverter>>(
      &getContext(), converter);
  mlir::concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                          converter);
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

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
std::unique_ptr<OperationPass<ModuleOp>> createConvertMidLFHEToLowLFHEPass() {
  return std::make_unique<MidLFHEToLowLFHEPass>();
}
} // namespace concretelang
} // namespace mlir

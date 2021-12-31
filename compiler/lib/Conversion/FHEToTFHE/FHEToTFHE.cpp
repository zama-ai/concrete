// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license
// information.

#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/FHEToTFHE/Patterns.h"
#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace {
struct FHEToTFHEPass : public FHEToTFHEBase<FHEToTFHEPass> {
  void runOnOperation() final;
};
} // namespace

using mlir::concretelang::FHE::EncryptedIntegerType;
using mlir::concretelang::TFHE::GLWECipherTextType;

/// FHEToTFHETypeConverter is a TypeConverter that transform
/// `FHE.eint<p>` to `TFHE.glwe<{_,_,_}{p}>`
class FHEToTFHETypeConverter : public mlir::TypeConverter {

public:
  FHEToTFHETypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([](EncryptedIntegerType type) {
      return mlir::concretelang::convertTypeEncryptedIntegerToGLWE(
          type.getContext(), type);
    });
    addConversion([](mlir::RankedTensorType type) {
      auto eint =
          type.getElementType().dyn_cast_or_null<EncryptedIntegerType>();
      if (eint == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(),
          mlir::concretelang::convertTypeEncryptedIntegerToGLWE(
              eint.getContext(), eint));
      return r;
    });
  }
};

void FHEToTFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  FHEToTFHETypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::concretelang::TFHE::TFHEDialect>();

  // Make sure that no ops from `FHE` remain after the lowering
  target.addIllegalDialect<mlir::concretelang::FHE::FHEDialect>();

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
  // Add all patterns required to lower all ops from `FHE` to
  // `TFHE`
  mlir::OwningRewritePatternList patterns(&getContext());

  populateWithGeneratedFHEToTFHE(patterns);
  patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                            FHEToTFHETypeConverter>>(
      &getContext(), converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                            FHEToTFHETypeConverter>>(
      &getContext(), converter);
  patterns.add<
      RegionOpTypeConverterPattern<mlir::scf::ForOp, FHEToTFHETypeConverter>>(
      &getContext(), converter);

  mlir::concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                              converter);
  mlir::populateFuncOpTypeConversionPattern(patterns, converter);

  // Conversion of RT Dialect Ops
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<
      mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(),
                                               converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowTaskOp>(target, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertFHEToTFHEPass() {
  return std::make_unique<FHEToTFHEPass>();
}
} // namespace concretelang
} // namespace mlir

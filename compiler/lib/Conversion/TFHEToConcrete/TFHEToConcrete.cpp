// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/TFHEToConcrete/Patterns.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace TFHE = mlir::concretelang::TFHE;
namespace Concrete = mlir::concretelang::Concrete;

namespace {
struct TFHEToConcretePass : public TFHEToConcreteBase<TFHEToConcretePass> {
  void runOnOperation() final;
};
} // namespace

using mlir::concretelang::Concrete::LweCiphertextType;
using mlir::concretelang::TFHE::GLWECipherTextType;

/// TFHEToConcreteTypeConverter is a TypeConverter that transform
/// `TFHE.glwe<{_,_,_}{p}>` to Concrete.lwe_ciphertext
class TFHEToConcreteTypeConverter : public mlir::TypeConverter {

public:
  TFHEToConcreteTypeConverter() {
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

namespace {

struct GLWEFromTableOpPattern
    : public mlir::OpRewritePattern<TFHE::GLWEFromTableOp> {
  GLWEFromTableOpPattern(mlir::MLIRContext *context,
                         mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<TFHE::GLWEFromTableOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::GLWEFromTableOp glweOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto oldTy = glweOp.getType().cast<TFHE::GLWECipherTextType>();
    auto newTy = rewriter.getType<Concrete::GlweCiphertextType>(
        oldTy.getDimension(), oldTy.getPolynomialSize(), oldTy.getP());

    rewriter.replaceOpWithNewOp<Concrete::GlweFromTable>(glweOp, newTy,
                                                         glweOp.table());
    return ::mlir::success();
  };
};

struct BootstrapGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::BootstrapGLWEOp> {
  BootstrapGLWEOpPattern(mlir::MLIRContext *context,
                         mlir::TypeConverter &converter,
                         mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<TFHE::BootstrapGLWEOp>(context, benefit),
        converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::BootstrapGLWEOp bsOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type resultType = converter.convertType(bsOp.getType());

    auto newOp = rewriter.replaceOpWithNewOp<Concrete::BootstrapLweOp>(
        bsOp, resultType, bsOp.ciphertext(), bsOp.lookup_table(), -1, -1,
        bsOp.level(), bsOp.baseLog());

    rewriter.startRootUpdate(newOp);

    newOp.input_ciphertext().setType(
        converter.convertType(bsOp.ciphertext().getType()));

    auto oldTy = bsOp.lookup_table().getType().cast<TFHE::GLWECipherTextType>();
    auto newTy = rewriter.getType<Concrete::GlweCiphertextType>(
        oldTy.getDimension(), oldTy.getPolynomialSize(), oldTy.getP());
    newOp.accumulator().setType(newTy);

    rewriter.finalizeRootUpdate(newOp);
    return ::mlir::success();
  }

private:
  mlir::TypeConverter &converter;
};

void TFHEToConcretePass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  TFHEToConcreteTypeConverter converter;

  // Mark ops from the target dialect as legal operations
  target.addLegalDialect<mlir::concretelang::Concrete::ConcreteDialect>();

  // Make sure that no ops from `TFHE` remain after the lowering
  target.addIllegalDialect<mlir::concretelang::TFHE::TFHEDialect>();

  // Make sure that no ops `linalg.generic` that have illegal types
  target.addDynamicallyLegalOp<mlir::linalg::GenericOp,
                               mlir::tensor::GenerateOp, mlir::scf::ForOp>(
      [&](mlir::Operation *op) {
        return (converter.isLegal(op->getOperandTypes()) &&
                converter.isLegal(op->getResultTypes()) &&
                converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
      });

  // Make sure that func has legal signature
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp funcOp) {
        return converter.isSignatureLegal(funcOp.getFunctionType()) &&
               converter.isLegal(&funcOp.getBody());
      });
  // Add all patterns required to lower all ops from `TFHE` to
  // `Concrete`
  mlir::RewritePatternSet patterns(&getContext());

  populateWithGeneratedTFHEToConcrete(patterns);

  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      mlir::concretelang::TFHE::ZeroTensorGLWEOp,
      mlir::concretelang::Concrete::ZeroTensorLWEOp>>(&getContext(), converter);
  patterns.add<GLWEFromTableOpPattern>(&getContext());
  patterns.add<BootstrapGLWEOpPattern>(&getContext(), converter);
  target.addDynamicallyLegalOp<Concrete::BootstrapLweOp>(
      [&](Concrete::BootstrapLweOp op) {
        return (converter.isLegal(op->getOperandTypes()) &&
                converter.isLegal(op->getResultTypes()));
      });
  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      TFHE::KeySwitchGLWEOp, Concrete::KeySwitchLweOp>>(&getContext(),
                                                        converter);
  patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                            TFHEToConcreteTypeConverter>>(
      &getContext(), converter);

  patterns.add<
      mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>>(
      patterns.getContext(), converter);

  patterns.add<
      mlir::concretelang::GenericTypeConverterPattern<mlir::linalg::YieldOp>>(
      patterns.getContext(), converter);

  patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                            TFHEToConcreteTypeConverter>>(
      &getContext(), converter);

  patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                            TFHEToConcreteTypeConverter>>(
      &getContext(), converter);
  mlir::concretelang::populateWithTensorTypeConverterPatterns(patterns, target,
                                                              converter);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);

  // Conversion of RT Dialect Ops
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<
      mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(),
                                               converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowTaskOp>(target, converter);
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<
      mlir::concretelang::RT::DataflowYieldOp>>(patterns.getContext(),
                                                converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DataflowYieldOp>(target, converter);

  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
      target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::linalg::YieldOp>(
      target, converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTFHEToConcretePass() {
  return std::make_unique<TFHEToConcretePass>();
}
} // namespace concretelang
} // namespace mlir

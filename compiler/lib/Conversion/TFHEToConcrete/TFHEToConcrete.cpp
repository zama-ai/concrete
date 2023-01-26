// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/TFHEToConcrete/Patterns.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
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
    addConversion([&](mlir::concretelang::RT::FutureType type) {
      return mlir::concretelang::RT::FutureType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::FutureType>()
                                .getElementType()));
    });
    addConversion([&](mlir::concretelang::RT::PointerType type) {
      return mlir::concretelang::RT::PointerType::get(
          this->convertType(type.dyn_cast<mlir::concretelang::RT::PointerType>()
                                .getElementType()));
    });
  }
};

namespace {

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
        bsOp, resultType, bsOp.ciphertext(), bsOp.lookup_table(), bsOp.level(),
        bsOp.baseLog(), bsOp.polySize(), bsOp.glweDimension());

    rewriter.startRootUpdate(newOp);
    newOp.input_ciphertext().setType(
        converter.convertType(bsOp.ciphertext().getType()));
    rewriter.finalizeRootUpdate(newOp);

    return ::mlir::success();
  }

private:
  mlir::TypeConverter &converter;
};

struct WopPBSGLWEOpPattern : public mlir::OpRewritePattern<TFHE::WopPBSGLWEOp> {
  WopPBSGLWEOpPattern(mlir::MLIRContext *context,
                      mlir::TypeConverter &converter,
                      mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<TFHE::WopPBSGLWEOp>(context, benefit),
        converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::WopPBSGLWEOp wopOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Type resultType = converter.convertType(wopOp.getType());

    auto newOp = rewriter.replaceOpWithNewOp<Concrete::WopPBSLweOp>(
        wopOp, resultType, wopOp.ciphertexts(), wopOp.lookupTable(),
        // Bootstrap parameters
        wopOp.bootstrapLevel(), wopOp.bootstrapBaseLog(),
        // Keyswitch parameters
        wopOp.keyswitchLevel(), wopOp.keyswitchBaseLog(),
        // Packing keyswitch key parameters
        wopOp.packingKeySwitchInputLweDimension(),
        wopOp.packingKeySwitchoutputPolynomialSize(),
        wopOp.packingKeySwitchLevel(), wopOp.packingKeySwitchBaseLog(),
        // Circuit bootstrap parameters
        wopOp.circuitBootstrapLevel(), wopOp.circuitBootstrapBaseLog(),
        // Crt Decomposition
        wopOp.crtDecomposition());

    rewriter.startRootUpdate(newOp);

    newOp.ciphertexts().setType(
        converter.convertType(wopOp.ciphertexts().getType()));

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

  // Legalize arith.constant operations introduced by some patterns
  target.addLegalOp<mlir::arith::ConstantOp>();

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
  target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
      [&](mlir::func::ConstantOp op) {
        return FunctionConstantOpConversion<
            TFHEToConcreteTypeConverter>::isLegal(op, converter);
      });
  // Add all patterns required to lower all ops from `TFHE` to
  // `Concrete`
  mlir::RewritePatternSet patterns(&getContext());

  patterns.add<FunctionConstantOpConversion<TFHEToConcreteTypeConverter>>(
      &getContext(), converter);
  populateWithGeneratedTFHEToConcrete(patterns);

  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      mlir::concretelang::TFHE::ZeroTensorGLWEOp,
      mlir::concretelang::Concrete::ZeroTensorLWEOp>>(&getContext(), converter);
  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      mlir::concretelang::TFHE::EncodeExpandLutForBootstrapOp,
      mlir::concretelang::Concrete::EncodeExpandLutForBootstrapOp>>(
      &getContext(), converter);
  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      mlir::concretelang::TFHE::EncodeExpandLutForWopPBSOp,
      mlir::concretelang::Concrete::EncodeExpandLutForWopPBSOp>>(&getContext(),
                                                                 converter);
  patterns.add<mlir::concretelang::GenericTypeAndOpConverterPattern<
      mlir::concretelang::TFHE::EncodePlaintextWithCrtOp,
      mlir::concretelang::Concrete::EncodePlaintextWithCrtOp>>(&getContext(),
                                                               converter);
  patterns.add<BootstrapGLWEOpPattern>(&getContext(), converter);
  patterns.add<WopPBSGLWEOpPattern>(&getContext(), converter);
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
  patterns.add<
      mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>,
      mlir::concretelang::GenericTypeConverterPattern<mlir::scf::YieldOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::bufferization::AllocTensorOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::MakeReadyFutureOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::AwaitFutureOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::CreateAsyncTaskOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::WorkFunctionReturnOp>,
      mlir::concretelang::GenericTypeConverterPattern<
          mlir::concretelang::RT::RegisterTaskWorkFunctionOp>>(&getContext(),
                                                               converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::MakeReadyFutureOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::AwaitFutureOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::CreateAsyncTaskOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>(
      target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::WorkFunctionReturnOp>(target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::concretelang::RT::RegisterTaskWorkFunctionOp>(target, converter);

  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
      target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::linalg::YieldOp>(
      target, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<
      mlir::bufferization::AllocTensorOp>(target, converter);

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

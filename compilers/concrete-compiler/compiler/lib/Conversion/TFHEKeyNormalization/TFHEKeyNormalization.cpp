// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/SmallSet.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h"
#include "concretelang/Conversion/Utils/RTOpConverter.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "concretelang/Support/Constants.h"
#include "concretelang/Support/TFHECircuitKeys.h"
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <variant>

namespace TFHE = mlir::concretelang::TFHE;

using mlir::concretelang::TFHE::GLWECipherTextType;

namespace conversion {

class KeyConverter {

public:
  KeyConverter(mlir::concretelang::TFHE::TFHECircuitKeys &circuitKeys)
      : circuitKeys(circuitKeys){};

  TFHE::GLWESecretKey convertSecretKey(TFHE::GLWESecretKey sk) {
    auto parameterizedKey = sk.getParameterized().value();
    return TFHE::GLWESecretKey::newNormalized(
        parameterizedKey.dimension, parameterizedKey.polySize,
        circuitKeys.getSecretKeyIndex(sk).value());
  }

  TFHE::GLWEBootstrapKeyAttr
  convertBootstrapKey(TFHE::GLWEBootstrapKeyAttr bsk) {
    return TFHE::GLWEBootstrapKeyAttr::get(
        bsk.getContext(), convertSecretKey(bsk.getInputKey()),
        convertSecretKey(bsk.getOutputKey()), bsk.getPolySize(),
        bsk.getGlweDim(), bsk.getLevels(), bsk.getBaseLog(),
        circuitKeys.getBootstrapKeyIndex(bsk).value());
  }

  TFHE::GLWEKeyswitchKeyAttr
  convertKeyswitchKey(TFHE::GLWEKeyswitchKeyAttr ksk) {
    return TFHE::GLWEKeyswitchKeyAttr::get(
        ksk.getContext(), convertSecretKey(ksk.getInputKey()),
        convertSecretKey(ksk.getOutputKey()), ksk.getLevels(), ksk.getBaseLog(),
        circuitKeys.getKeyswitchKeyIndex(ksk).value());
  }

  TFHE::GLWEPackingKeyswitchKeyAttr
  convertPackingKeyswitchKey(TFHE::GLWEPackingKeyswitchKeyAttr pksk) {
    return TFHE::GLWEPackingKeyswitchKeyAttr::get(
        pksk.getContext(), convertSecretKey(pksk.getInputKey()),
        convertSecretKey(pksk.getOutputKey()), pksk.getOutputPolySize(),
        pksk.getInnerLweDim(), pksk.getGlweDim(), pksk.getLevels(),
        pksk.getBaseLog(),
        circuitKeys.getPackingKeyswitchKeyIndex(pksk).value());
  }

private:
  mlir::concretelang::TFHE::TFHECircuitKeys circuitKeys;
};

class TypeConverter : public mlir::TypeConverter {

public:
  TypeConverter(KeyConverter &keyConverter) : keyConverter(keyConverter) {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      auto key = type.getKey();
      if (key.isParameterized()) {
        return GLWECipherTextType::get(type.getContext(),
                                       keyConverter.convertSecretKey(key));
      } else {
        return type;
      }
    });
    addConversion([&](mlir::RankedTensorType type) {
      mlir::Type r = mlir::RankedTensorType::get(
          type.getShape(), this->convertType(type.getElementType()));
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

private:
  KeyConverter keyConverter;
};
} // namespace conversion

namespace patterns {
struct KeySwitchGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::KeySwitchGLWEOp> {
  KeySwitchGLWEOpPattern(mlir::MLIRContext *context,
                         conversion::TypeConverter &typeConverter,
                         conversion::KeyConverter &keyConverter,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::KeySwitchGLWEOp>(context, benefit),
        keyConverter(keyConverter), typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::KeySwitchGLWEOp ksOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto newInputTy = typeConverter.convertType(ksOp.getCiphertext().getType())
                          .cast<GLWECipherTextType>();
    auto newOutputTy = typeConverter.convertType(ksOp.getResult().getType());
    auto newKeyswitchKey = keyConverter.convertKeyswitchKey(ksOp.getKeyAttr());
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::KeySwitchGLWEOp>(
        ksOp, newOutputTy, ksOp.getCiphertext(), newKeyswitchKey);
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertext().setType(newInputTy);
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  conversion::KeyConverter &keyConverter;
  conversion::TypeConverter &typeConverter;
};

struct BatchedKeySwitchGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::BatchedKeySwitchGLWEOp> {
  BatchedKeySwitchGLWEOpPattern(mlir::MLIRContext *context,
                                conversion::TypeConverter &typeConverter,
                                conversion::KeyConverter &keyConverter,
                                mlir::PatternBenefit benefit =
                                    mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::BatchedKeySwitchGLWEOp>(context, benefit),
        keyConverter(keyConverter), typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::BatchedKeySwitchGLWEOp ksOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto newInputTy = typeConverter.convertType(ksOp.getCiphertexts().getType())
                          .cast<mlir::TensorType>();
    auto newOutputTy = typeConverter.convertType(ksOp.getResult().getType());
    auto newKeyswitchKey = keyConverter.convertKeyswitchKey(ksOp.getKeyAttr());
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::BatchedKeySwitchGLWEOp>(
        ksOp, newOutputTy, ksOp.getCiphertexts(), newKeyswitchKey);
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertexts().setType(newInputTy);
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  conversion::KeyConverter &keyConverter;
  conversion::TypeConverter &typeConverter;
};

struct BootstrapGLWEOpPattern
    : public mlir::OpRewritePattern<TFHE::BootstrapGLWEOp> {
  BootstrapGLWEOpPattern(mlir::MLIRContext *context,
                         conversion::TypeConverter &typeConverter,
                         conversion::KeyConverter &keyConverter,
                         mlir::PatternBenefit benefit =
                             mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::BootstrapGLWEOp>(context, benefit),
        keyConverter(keyConverter), typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::BootstrapGLWEOp bsOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto newInputTy = typeConverter.convertType(bsOp.getCiphertext().getType())
                          .cast<GLWECipherTextType>();
    auto newOutputTy = typeConverter.convertType(bsOp.getResult().getType());
    auto newBootstrapKey = keyConverter.convertBootstrapKey(bsOp.getKeyAttr());
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::BootstrapGLWEOp>(
        bsOp, newOutputTy, bsOp.getCiphertext(), bsOp.getLookupTable(),
        newBootstrapKey);
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertext().setType(newInputTy.cast<GLWECipherTextType>());
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  conversion::KeyConverter &keyConverter;
  conversion::TypeConverter &typeConverter;
};

struct WopPBSGLWEOpPattern : public mlir::OpRewritePattern<TFHE::WopPBSGLWEOp> {
  WopPBSGLWEOpPattern(mlir::MLIRContext *context,
                      conversion::TypeConverter &typeConverter,
                      conversion::KeyConverter &keyConverter,
                      mlir::PatternBenefit benefit =
                          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<TFHE::WopPBSGLWEOp>(context, benefit),
        keyConverter(keyConverter), typeConverter(typeConverter) {}

  mlir::LogicalResult
  matchAndRewrite(TFHE::WopPBSGLWEOp wopPBSOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto newInputTy =
        typeConverter.convertType(wopPBSOp.getCiphertexts().getType())
            .cast<mlir::RankedTensorType>();
    auto newOutputType = typeConverter.convertType(wopPBSOp.getType());
    auto newKeyswitchKey =
        keyConverter.convertKeyswitchKey(wopPBSOp.getKskAttr());
    auto newBootstrapKey =
        keyConverter.convertBootstrapKey(wopPBSOp.getBskAttr());
    auto newPackingKeyswitchKey =
        keyConverter.convertPackingKeyswitchKey(wopPBSOp.getPkskAttr());
    auto newOp = rewriter.replaceOpWithNewOp<TFHE::WopPBSGLWEOp>(
        wopPBSOp, newOutputType, wopPBSOp.getCiphertexts(),
        wopPBSOp.getLookupTable(), newKeyswitchKey, newBootstrapKey,
        newPackingKeyswitchKey, wopPBSOp.getCrtDecompositionAttr(),
        wopPBSOp.getCbsLevelsAttr(), wopPBSOp.getCbsBaseLogAttr());
    rewriter.startRootUpdate(newOp);
    newOp.getCiphertexts().setType(newInputTy);
    rewriter.finalizeRootUpdate(newOp);
    return mlir::success();
  };

private:
  conversion::KeyConverter &keyConverter;
  conversion::TypeConverter &typeConverter;
};
} // namespace patterns

namespace {
struct TFHEKeyNormalizationPass
    : public TFHEKeyNormalizationBase<TFHEKeyNormalizationPass> {
  void runOnOperation() final;
};

template <typename Op>
void populateWithTFHEOpTypeConversionPattern(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter) {
  patterns.add<mlir::concretelang::GenericTypeConverterPattern<Op>>(
      patterns.getContext(), typeConverter);

  target.addDynamicallyLegalOp<Op>(
      [&](Op op) { return typeConverter.isLegal(op->getResultTypes()); });
}

/// Populate the RewritePatternSet with all patterns that rewrite Concrete
/// operators to the corresponding function call to the `Concrete C API`.
void populateWithTFHEOpTypeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::ConversionTarget &target,
    mlir::TypeConverter &typeConverter) {
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::ZeroGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::ZeroTensorGLWEOp>(patterns, target,
                                                  typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::AddGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::AddGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::SubGLWEIntOp>(patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<mlir::concretelang::TFHE::NegGLWEOp>(
      patterns, target, typeConverter);
  populateWithTFHEOpTypeConversionPattern<
      mlir::concretelang::TFHE::MulGLWEIntOp>(patterns, target, typeConverter);
}
} // namespace

void TFHEKeyNormalizationPass::runOnOperation() {
  auto op = this->getOperation();

  auto circuitKeys = TFHE::extractCircuitKeys(op);
  auto keyConverter = conversion::KeyConverter(circuitKeys);
  auto typeConverter = conversion::TypeConverter(keyConverter);

  // Parametrize
  {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    // function signature
    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return typeConverter.isSignatureLegal(funcOp.getFunctionType()) &&
                 typeConverter.isLegal(&funcOp.getBody());
        });
    target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
        [&](mlir::func::ConstantOp op) {
          return FunctionConstantOpConversion<
              conversion::TypeConverter>::isLegal(op, typeConverter);
        });
    patterns.add<FunctionConstantOpConversion<conversion::TypeConverter>>(
        &getContext(), typeConverter);
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, typeConverter);

    // Parametrize keyswitch
    target.addLegalOp<mlir::arith::ConstantOp>();
    patterns.add<patterns::KeySwitchGLWEOpPattern>(&getContext(), typeConverter,
                                                   keyConverter);
    target.addDynamicallyLegalOp<TFHE::KeySwitchGLWEOp>(
        [&](TFHE::KeySwitchGLWEOp op) {
          return op.getKeyAttr().getInputKey().isNormalized() &&
                 op.getKeyAttr().getOutputKey().isNormalized() &&
                 op.getKeyAttr().getIndex() != -1;
        });

    patterns.add<patterns::BatchedKeySwitchGLWEOpPattern>(
        &getContext(), typeConverter, keyConverter);
    target.addDynamicallyLegalOp<TFHE::BatchedKeySwitchGLWEOp>(
        [&](TFHE::BatchedKeySwitchGLWEOp op) {
          return op.getKeyAttr().getInputKey().isNormalized() &&
                 op.getKeyAttr().getOutputKey().isNormalized() &&
                 op.getKeyAttr().getIndex() != -1;
        });

    // Parametrize bootstrap
    patterns.add<patterns::BootstrapGLWEOpPattern>(&getContext(), typeConverter,
                                                   keyConverter);
    target.addDynamicallyLegalOp<TFHE::BootstrapGLWEOp>(
        [&](TFHE::BootstrapGLWEOp op) {
          return op.getKeyAttr().getInputKey().isNormalized() &&
                 op.getKeyAttr().getOutputKey().isNormalized() &&
                 op.getKeyAttr().getIndex() != -1;
        });

    // Parametrize wop pbs
    patterns.add<patterns::WopPBSGLWEOpPattern>(&getContext(), typeConverter,
                                                keyConverter);
    target.addDynamicallyLegalOp<TFHE::WopPBSGLWEOp>(
        [&](TFHE::WopPBSGLWEOp op) {
          return op.getKskAttr().getInputKey().isNormalized() &&
                 op.getKskAttr().getOutputKey().isNormalized() &&
                 op.getKskAttr().getIndex() != -1 &&
                 op.getBskAttr().getInputKey().isNormalized() &&
                 op.getBskAttr().getOutputKey().isNormalized() &&
                 op.getBskAttr().getIndex() != -1 &&
                 op.getPkskAttr().getInputKey().isNormalized() &&
                 op.getPkskAttr().getOutputKey().isNormalized() &&
                 op.getPkskAttr().getIndex() != -1;
        });

    // Add all patterns to convert TFHE types
    populateWithTFHEOpTypeConversionPatterns(patterns, target, typeConverter);

    patterns.add<mlir::concretelang::GenericTypeConverterPattern<
        mlir::bufferization::AllocTensorOp>>(&getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::bufferization::AllocTensorOp>(target, typeConverter);

    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<mlir::tensor::EmptyOp>>(
        &getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::EmptyOp>(
        target, typeConverter);

    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<mlir::tensor::DimOp>>(
        &getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::DimOp>(
        target, typeConverter);

    patterns.add<mlir::concretelang::GenericTypeConverterPattern<
        mlir::tensor::ParallelInsertSliceOp>>(&getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::tensor::ParallelInsertSliceOp>(target, typeConverter);

    patterns.add<RegionOpTypeConverterPattern<mlir::linalg::GenericOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    patterns.add<RegionOpTypeConverterPattern<mlir::tensor::GenerateOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForallOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    patterns.add<RegionOpTypeConverterPattern<mlir::scf::InParallelOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    patterns.add<RegionOpTypeConverterPattern<mlir::func::ReturnOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
        target, typeConverter);
    patterns.add<RegionOpTypeConverterPattern<mlir::linalg::YieldOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::linalg::YieldOp>(
        target, typeConverter);

    patterns.add<RegionOpTypeConverterPattern<mlir::tensor::YieldOp,
                                              conversion::TypeConverter>>(
        &getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<mlir::tensor::YieldOp>(
        target, typeConverter);

    mlir::concretelang::populateWithTensorTypeConverterPatterns(
        patterns, target, typeConverter);

    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::Tracing::TraceCiphertextOp>,
        mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<mlir::scf::YieldOp>>(
        &getContext(), typeConverter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::Tracing::TraceCiphertextOp>(target, typeConverter);

    mlir::concretelang::populateWithRTTypeConverterPatterns(patterns, target,
                                                            typeConverter);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createTFHEKeyNormalizationPass() {
  return std::make_unique<TFHEKeyNormalizationPass>();
}
} // namespace concretelang
} // namespace mlir

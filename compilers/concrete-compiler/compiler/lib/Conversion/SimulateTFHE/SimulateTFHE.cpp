// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/RTOpConverter.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Conversion/Utils/Utils.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Support/Constants.h"

namespace TFHE = mlir::concretelang::TFHE;
namespace Tracing = mlir::concretelang::Tracing;

using TFHE::GLWECipherTextType;

/// Converts ciphertexts to plaintext integer types
class SimulateTFHETypeConverter : public mlir::TypeConverter {

public:
  SimulateTFHETypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto glwe = type.getElementType().dyn_cast_or_null<GLWECipherTextType>();
      if (glwe == nullptr) {
        return (mlir::Type)(type);
      }
      return (mlir::Type)mlir::RankedTensorType::get(
          type.getShape(), mlir::IntegerType::get(type.getContext(), 64));
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

mlir::RankedTensorType toDynamicTensorType(mlir::TensorType staticSizedTensor) {
  std::vector<int64_t> dynSizedShape(staticSizedTensor.getShape().size(),
                                     mlir::ShapedType::kDynamic);
  return mlir::RankedTensorType::get(dynSizedShape,
                                     staticSizedTensor.getElementType());
}

struct NegOpPattern : public mlir::OpConversionPattern<TFHE::NegGLWEOp> {

  NegOpPattern(mlir::MLIRContext *context, mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::NegGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::NegGLWEOp negOp, TFHE::NegGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_neg_lwe_u64";

    if (insertForwardDeclaration(
            negOp, rewriter, funcName,
            rewriter.getFunctionType({rewriter.getIntegerType(64)},
                                     {rewriter.getIntegerType(64)}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        negOp, funcName, mlir::TypeRange{rewriter.getIntegerType(64)},
        mlir::ValueRange({adaptor.getA()}));

    return mlir::success();
  }
};

template <typename AddOp, typename AddOpAdaptor>
struct AddOpPattern : public mlir::OpConversionPattern<AddOp> {

  AddOpPattern(mlir::MLIRContext *context, mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<AddOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(AddOp addOp, AddOpAdaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_add_lwe_u64";

    if (insertForwardDeclaration(
            addOp, rewriter, funcName,
            rewriter.getFunctionType(
                {rewriter.getIntegerType(64), rewriter.getIntegerType(64)},
                {rewriter.getIntegerType(64)}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        addOp, funcName, mlir::TypeRange{rewriter.getIntegerType(64)},
        mlir::ValueRange({adaptor.getA(), adaptor.getB()}));

    return mlir::success();
  }
};

struct MulOpPattern : public mlir::OpConversionPattern<TFHE::MulGLWEIntOp> {

  MulOpPattern(mlir::MLIRContext *context, mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::MulGLWEIntOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::MulGLWEIntOp mulOp, TFHE::MulGLWEIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_mul_lwe_u64";

    if (insertForwardDeclaration(
            mulOp, rewriter, funcName,
            rewriter.getFunctionType(
                {rewriter.getIntegerType(64), rewriter.getIntegerType(64)},
                {rewriter.getIntegerType(64)}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        mulOp, funcName, mlir::TypeRange{rewriter.getIntegerType(64)},
        mlir::ValueRange({adaptor.getA(), adaptor.getB()}));

    return mlir::success();
  }
};

struct SubIntGLWEOpPattern : public mlir::OpRewritePattern<TFHE::SubGLWEIntOp> {

  SubIntGLWEOpPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<TFHE::SubGLWEIntOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::SubGLWEIntOp subOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value negated = rewriter.create<TFHE::NegGLWEOp>(
        subOp.getLoc(), subOp.getB().getType(), subOp.getB());

    rewriter.replaceOpWithNewOp<TFHE::AddGLWEIntOp>(subOp, subOp.getType(),
                                                    negated, subOp.getA());

    return mlir::success();
  }
};

struct EncodeExpandLutForBootstrapOpPattern
    : public mlir::OpConversionPattern<TFHE::EncodeExpandLutForBootstrapOp> {

  EncodeExpandLutForBootstrapOpPattern(mlir::MLIRContext *context,
                                       mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::EncodeExpandLutForBootstrapOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::EncodeExpandLutForBootstrapOp eeOp,
                  TFHE::EncodeExpandLutForBootstrapOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_encode_expand_lut_for_boostrap";

    mlir::Value polySizeCst = rewriter.create<mlir::arith::ConstantIntOp>(
        eeOp.getLoc(), eeOp.getPolySize(), 32);
    mlir::Value outputBitsCst = rewriter.create<mlir::arith::ConstantIntOp>(
        eeOp.getLoc(), eeOp.getOutputBits(), 32);
    mlir::Value isSignedCst = rewriter.create<mlir::arith::ConstantIntOp>(
        eeOp.getLoc(), eeOp.getIsSigned(), 1);

    mlir::Value outputBuffer =
        rewriter.create<mlir::bufferization::AllocTensorOp>(
            eeOp.getLoc(),
            eeOp.getResult().getType().cast<mlir::RankedTensorType>(),
            mlir::ValueRange{});

    auto dynamicResultType = toDynamicTensorType(eeOp.getResult().getType());
    auto dynamicLutType =
        toDynamicTensorType(eeOp.getInputLookupTable().getType());

    mlir::Value castedOutputBuffer = rewriter.create<mlir::tensor::CastOp>(
        eeOp.getLoc(), dynamicResultType, outputBuffer);

    mlir::Value castedLUT = rewriter.create<mlir::tensor::CastOp>(
        eeOp.getLoc(),
        toDynamicTensorType(eeOp.getInputLookupTable().getType()),
        adaptor.getInputLookupTable());

    // sim_encode_expand_lut_for_boostrap(uint64_t *out_allocated, uint64_t
    // *out_aligned, uint64_t out_offset, uint64_t out_size, uint64_t
    // out_stride, uint64_t *in_allocated, uint64_t *in_aligned, uint64_t
    // in_offset, uint64_t in_size, uint64_t in_stride, uint32_t poly_size,
    // uint32_t output_bits, bool is_signed)
    if (insertForwardDeclaration(
            eeOp, rewriter, funcName,
            rewriter.getFunctionType(
                {dynamicResultType, dynamicLutType, rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(1)},
                {}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(
        eeOp.getLoc(), funcName, mlir::TypeRange{},
        mlir::ValueRange({castedOutputBuffer, castedLUT, polySizeCst,
                          outputBitsCst, isSignedCst}));

    rewriter.replaceOp(eeOp, outputBuffer);

    return mlir::success();
  }
};

struct EncodeLutForCrtWopPBSOpPattern
    : public mlir::OpConversionPattern<TFHE::EncodeLutForCrtWopPBSOp> {

  EncodeLutForCrtWopPBSOpPattern(mlir::MLIRContext *context,
                                 mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::EncodeLutForCrtWopPBSOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::EncodeLutForCrtWopPBSOp encodeOp,
                  TFHE::EncodeLutForCrtWopPBSOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_encode_lut_for_crt_woppbs";

    mlir::Value modulusProductCst = rewriter.create<mlir::arith::ConstantIntOp>(
        encodeOp.getLoc(), encodeOp.getModulusProduct(), 32);
    mlir::Value isSignedCst = rewriter.create<mlir::arith::ConstantIntOp>(
        encodeOp.getLoc(), encodeOp.getIsSigned(), 1);

    mlir::Value outputBuffer =
        rewriter.create<mlir::bufferization::AllocTensorOp>(
            encodeOp.getLoc(),
            encodeOp.getResult().getType().cast<mlir::RankedTensorType>(),
            mlir::ValueRange{});

    auto dynamicResultType =
        toDynamicTensorType(encodeOp.getResult().getType());
    auto dynamicLutType =
        toDynamicTensorType(encodeOp.getInputLookupTable().getType());

    mlir::Value castedOutputBuffer = rewriter.create<mlir::tensor::CastOp>(
        encodeOp.getLoc(), dynamicResultType, outputBuffer);
    mlir::Value castedLUT = rewriter.create<mlir::tensor::CastOp>(
        encodeOp.getLoc(), dynamicLutType, adaptor.getInputLookupTable());

    auto crtDecompValue = mlir::concretelang::globalMemrefFromArrayAttr(
        rewriter, encodeOp.getLoc(), encodeOp.getCrtDecompositionAttr());
    auto crtBitsValue = mlir::concretelang::globalMemrefFromArrayAttr(
        rewriter, encodeOp.getLoc(), encodeOp.getCrtBitsAttr());

    if (insertForwardDeclaration(
            encodeOp, rewriter, funcName,
            rewriter.getFunctionType(
                {dynamicResultType, dynamicLutType, crtDecompValue.getType(),
                 crtBitsValue.getType(), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(1)},
                {}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(
        encodeOp.getLoc(), funcName, mlir::TypeRange{},
        mlir::ValueRange({castedOutputBuffer, castedLUT, crtDecompValue,
                          crtBitsValue, modulusProductCst, isSignedCst}));

    rewriter.replaceOp(encodeOp, outputBuffer);

    return mlir::success();
  }
};

struct EncodePlaintextWithCrtOpPattern
    : public mlir::OpConversionPattern<TFHE::EncodePlaintextWithCrtOp> {

  EncodePlaintextWithCrtOpPattern(mlir::MLIRContext *context,
                                  mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::EncodePlaintextWithCrtOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::EncodePlaintextWithCrtOp epOp,
                  TFHE::EncodePlaintextWithCrtOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_encode_plaintext_with_crt";

    mlir::Value modsProductCst = rewriter.create<mlir::arith::ConstantIntOp>(
        epOp.getLoc(), epOp.getModsProd(), 64);

    mlir::Value outputBuffer =
        rewriter.create<mlir::bufferization::AllocTensorOp>(
            epOp.getLoc(),
            epOp.getResult().getType().cast<mlir::RankedTensorType>(),
            mlir::ValueRange{});

    auto dynamicResultType = toDynamicTensorType(epOp.getResult().getType());

    mlir::Value castedOutputBuffer = rewriter.create<mlir::tensor::CastOp>(
        epOp.getLoc(), dynamicResultType, outputBuffer);

    auto ModsValue = mlir::concretelang::globalMemrefFromArrayAttr(
        rewriter, epOp.getLoc(), epOp.getModsAttr());

    if (insertForwardDeclaration(
            epOp, rewriter, funcName,
            rewriter.getFunctionType(
                {dynamicResultType, epOp.getInput().getType(),
                 ModsValue.getType(), rewriter.getI64Type()},
                {}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(
        epOp.getLoc(), funcName, mlir::TypeRange{},
        mlir::ValueRange({castedOutputBuffer, adaptor.getInput(), ModsValue,
                          modsProductCst}));

    rewriter.replaceOp(epOp, outputBuffer);

    return mlir::success();
  }
};

struct WopPBSGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::WopPBSGLWEOp> {

  WopPBSGLWEOpPattern(mlir::MLIRContext *context,
                      mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::WopPBSGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::WopPBSGLWEOp wopPbs,
                  TFHE::WopPBSGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_wop_pbs_crt";

    auto resultType = wopPbs.getType().cast<mlir::RankedTensorType>();
    auto inputType =
        wopPbs.getCiphertexts().getType().cast<mlir::RankedTensorType>();

    mlir::Value outputBuffer =
        rewriter.create<mlir::bufferization::AllocTensorOp>(
            wopPbs.getLoc(),
            this->getTypeConverter()
                ->convertType(resultType)
                .cast<mlir::RankedTensorType>(),
            mlir::ValueRange{});

    auto dynamicResultType = toDynamicTensorType(this->getTypeConverter()
                                                     ->convertType(resultType)
                                                     .cast<mlir::TensorType>());
    auto dynamicInputType = toDynamicTensorType(this->getTypeConverter()
                                                    ->convertType(inputType)
                                                    .cast<mlir::TensorType>());
    auto dynamicLutType =
        toDynamicTensorType(wopPbs.getLookupTable().getType());

    mlir::Value castedOutputBuffer = rewriter.create<mlir::tensor::CastOp>(
        wopPbs.getLoc(), dynamicResultType, outputBuffer);
    mlir::Value castedCiphertexts = rewriter.create<mlir::tensor::CastOp>(
        wopPbs.getLoc(), dynamicInputType, adaptor.getCiphertexts());
    mlir::Value castedLut = rewriter.create<mlir::tensor::CastOp>(
        wopPbs.getLoc(), dynamicLutType, adaptor.getLookupTable());

    auto lweDimCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getPksk().getInnerLweDim(), 32);
    auto cbsLevelCountCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getCbsLevels(), 32);
    auto cbsBaseLogCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getCbsBaseLog(), 32);
    auto kskLevelCountCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getKsk().getLevels(), 32);
    auto kskBaseLogCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getKsk().getBaseLog(), 32);
    auto bskLevelCountCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getBsk().getLevels(), 32);
    auto bskBaseLogCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getBsk().getBaseLog(), 32);
    auto fpkskLevelCountCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getPksk().getLevels(), 32);
    auto fpkskBaseLogCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getPksk().getBaseLog(), 32);
    auto polySizeCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getPksk().getOutputPolySize(), 32);
    auto glweDimCst = rewriter.create<mlir::arith::ConstantIntOp>(
        wopPbs.getLoc(), adaptor.getBsk().getGlweDim(), 32);

    auto crtDecompValue = mlir::concretelang::globalMemrefFromArrayAttr(
        rewriter, wopPbs.getLoc(), wopPbs.getCrtDecompositionAttr());

    if (insertForwardDeclaration(
            wopPbs, rewriter, funcName,
            rewriter.getFunctionType(
                {dynamicResultType, dynamicInputType, dynamicLutType,
                 crtDecompValue.getType(), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32)},
                {}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.create<mlir::func::CallOp>(
        wopPbs.getLoc(), funcName, mlir::TypeRange{},
        mlir::ValueRange({castedOutputBuffer, castedCiphertexts, castedLut,
                          crtDecompValue, lweDimCst, cbsLevelCountCst,
                          cbsBaseLogCst, kskLevelCountCst, kskBaseLogCst,
                          bskLevelCountCst, bskBaseLogCst, polySizeCst,
                          fpkskLevelCountCst, fpkskBaseLogCst, glweDimCst}));

    rewriter.replaceOp(wopPbs, outputBuffer);

    return mlir::success();
  }
};

struct BootstrapGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::BootstrapGLWEOp> {

  BootstrapGLWEOpPattern(mlir::MLIRContext *context,
                         mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::BootstrapGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::BootstrapGLWEOp bsOp,
                  TFHE::BootstrapGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_bootstrap_lwe_u64";

    TFHE::GLWECipherTextType resultType =
        bsOp.getType().cast<TFHE::GLWECipherTextType>();
    TFHE::GLWECipherTextType inputType =
        bsOp.getCiphertext().getType().cast<TFHE::GLWECipherTextType>();

    auto polySize = adaptor.getKey().getPolySize();
    auto glweDimension = adaptor.getKey().getGlweDim();
    auto levels = adaptor.getKey().getLevels();
    auto baseLog = adaptor.getKey().getBaseLog();
    auto inputLweDimension =
        inputType.getKey().getNormalized().value().dimension;

    auto polySizeCst = rewriter.create<mlir::arith::ConstantIntOp>(
        bsOp.getLoc(), polySize, 32);
    auto glweDimensionCst = rewriter.create<mlir::arith::ConstantIntOp>(
        bsOp.getLoc(), glweDimension, 32);
    auto levelsCst =
        rewriter.create<mlir::arith::ConstantIntOp>(bsOp.getLoc(), levels, 32);
    auto baseLogCst =
        rewriter.create<mlir::arith::ConstantIntOp>(bsOp.getLoc(), baseLog, 32);
    auto inputLweDimensionCst = rewriter.create<mlir::arith::ConstantIntOp>(
        bsOp.getLoc(), inputLweDimension, 32);

    auto dynamicLutType = toDynamicTensorType(bsOp.getLookupTable().getType());

    mlir::Value castedLUT = rewriter.create<mlir::tensor::CastOp>(
        bsOp.getLoc(), dynamicLutType, adaptor.getLookupTable());

    // uint64_t sim_bootstrap_lwe_u64(uint64_t plaintext, uint64_t
    // *tlu_allocated, uint64_t *tlu_aligned, uint64_t tlu_offset, uint64_t
    // tlu_size, uint64_t tlu_stride, uint32_t input_lwe_dim, uint32_t
    // poly_size, uint32_t level, uint32_t base_log, uint32_t glwe_dim)
    if (insertForwardDeclaration(
            bsOp, rewriter, funcName,
            rewriter.getFunctionType(
                {rewriter.getIntegerType(64), dynamicLutType,
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32)},
                {rewriter.getIntegerType(64)}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        bsOp, funcName, this->getTypeConverter()->convertType(resultType),
        mlir::ValueRange({adaptor.getCiphertext(), castedLUT,
                          inputLweDimensionCst, polySizeCst, levelsCst,
                          baseLogCst, glweDimensionCst}));

    return mlir::success();
  }
};

struct KeySwitchGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::KeySwitchGLWEOp> {

  KeySwitchGLWEOpPattern(mlir::MLIRContext *context,
                         mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::KeySwitchGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::KeySwitchGLWEOp ksOp,
                  TFHE::KeySwitchGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    const std::string funcName = "sim_keyswitch_lwe_u64";

    TFHE::GLWECipherTextType resultType =
        ksOp.getType().cast<TFHE::GLWECipherTextType>();
    TFHE::GLWECipherTextType inputType =
        ksOp.getCiphertext().getType().cast<TFHE::GLWECipherTextType>();

    auto levels = adaptor.getKey().getLevels();
    auto baseLog = adaptor.getKey().getBaseLog();
    auto inputDim = inputType.getKey().getNormalized().value().dimension;
    auto outputDim = resultType.getKey().getNormalized().value().dimension;

    mlir::Value levelCst =
        rewriter.create<mlir::arith::ConstantIntOp>(ksOp.getLoc(), levels, 32);
    mlir::Value baseLogCst =
        rewriter.create<mlir::arith::ConstantIntOp>(ksOp.getLoc(), baseLog, 32);
    mlir::Value inputDimCst = rewriter.create<mlir::arith::ConstantIntOp>(
        ksOp.getLoc(), inputDim, 32);
    mlir::Value outputDimCst = rewriter.create<mlir::arith::ConstantIntOp>(
        ksOp.getLoc(), outputDim, 32);

    // uint64_t sim_keyswitch_lwe_u64(uint64_t plaintext, uint32_t level,
    // uint32_t base_log, uint32_t input_lwe_dim, uint32_t output_lwe_dim)
    if (insertForwardDeclaration(
            ksOp, rewriter, funcName,
            rewriter.getFunctionType(
                {rewriter.getIntegerType(64), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32), rewriter.getIntegerType(32),
                 rewriter.getIntegerType(32)},
                {rewriter.getIntegerType(64)}))
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        ksOp, funcName, this->getTypeConverter()->convertType(resultType),
        mlir::ValueRange({adaptor.getCiphertext(), levelCst, baseLogCst,
                          inputDimCst, outputDimCst}));

    return mlir::success();
  }
};

struct ZeroOpPattern : public mlir::OpConversionPattern<TFHE::ZeroGLWEOp> {
  ZeroOpPattern(mlir::MLIRContext *context, mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::ZeroGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::ZeroGLWEOp zeroOp, TFHE::ZeroGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newResultTy = this->getTypeConverter()->convertType(zeroOp.getType());
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantIntOp>(zeroOp, 0,
                                                            newResultTy);
    return ::mlir::success();
  };
};

struct ZeroTensorOpPattern
    : public mlir::OpConversionPattern<TFHE::ZeroTensorGLWEOp> {
  ZeroTensorOpPattern(mlir::MLIRContext *context,
                      mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::ZeroTensorGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::ZeroTensorGLWEOp zeroTensorOp,
                  TFHE::ZeroTensorGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newResultTy =
        this->getTypeConverter()->convertType(zeroTensorOp.getType());
    rewriter.replaceOpWithNewOp<mlir::arith::ConstantOp>(
        zeroTensorOp,
        mlir::DenseElementsAttr::get(newResultTy, {mlir::APInt::getZero(64)}),
        newResultTy);
    return ::mlir::success();
  };
};

struct SimulateTFHEPass : public SimulateTFHEBase<SimulateTFHEPass> {
  bool enableOverflowDetection;
  SimulateTFHEPass(bool enableOverflowDetection)
      : enableOverflowDetection(enableOverflowDetection) {}

  void runOnOperation() final;
};

void SimulateTFHEPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  SimulateTFHETypeConverter converter;

  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addLegalOp<mlir::func::CallOp, mlir::memref::GetGlobalOp,
                    mlir::memref::CastOp, mlir::bufferization::AllocTensorOp,
                    mlir::tensor::CastOp>();
  // Make sure that no ops from `TFHE` remain after the lowering
  target.addIllegalDialect<TFHE::TFHEDialect>();

  mlir::RewritePatternSet patterns(&getContext());

  // Convert operand and result types
  patterns.insert<mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::bufferization::AllocTensorOp, true>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::scf::YieldOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::FromElementsOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::ExtractOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::ExtractSliceOp, true>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::InsertOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::InsertSliceOp, true>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::ExpandShapeOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::CollapseShapeOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::YieldOp>,
                  mlir::concretelang::TypeConvertingReinstantiationPattern<
                      mlir::tensor::EmptyOp>>(&getContext(), converter);
  // legalize ops only if operand and result types are legal
  target.addDynamicallyLegalOp<
      mlir::tensor::YieldOp, mlir::scf::YieldOp, mlir::tensor::GenerateOp,
      mlir::tensor::ExtractSliceOp, mlir::tensor::ExtractOp,
      mlir::tensor::InsertOp, mlir::tensor::InsertSliceOp,
      mlir::tensor::FromElementsOp, mlir::tensor::ExpandShapeOp,
      mlir::tensor::CollapseShapeOp, mlir::bufferization::AllocTensorOp,
      mlir::tensor::EmptyOp>([&](mlir::Operation *op) {
    return converter.isLegal(op->getResultTypes()) &&
           converter.isLegal(op->getOperandTypes());
  });
  // Make sure that no ops `linalg.generic` that have illegal types
  target
      .addDynamicallyLegalOp<mlir::linalg::GenericOp, mlir::tensor::GenerateOp>(
          [&](mlir::Operation *op) {
            return (
                converter.isLegal(op->getOperandTypes()) &&
                converter.isLegal(op->getResultTypes()) &&
                converter.isLegal(op->getRegion(0).front().getArgumentTypes()));
          });
  // Update scf::ForOp region with converted types
  patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                            SimulateTFHETypeConverter>>(
      &getContext(), converter);
  target.addDynamicallyLegalOp<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
    return converter.isLegal(forOp.getInitArgs().getTypes()) &&
           converter.isLegal(forOp.getResults().getTypes());
  });

  patterns.insert<ZeroOpPattern, ZeroTensorOpPattern, KeySwitchGLWEOpPattern,
                  BootstrapGLWEOpPattern, WopPBSGLWEOpPattern,
                  EncodeExpandLutForBootstrapOpPattern,
                  EncodeLutForCrtWopPBSOpPattern,
                  EncodePlaintextWithCrtOpPattern, NegOpPattern>(&getContext(),
                                                                 converter);
  patterns.insert<SubIntGLWEOpPattern>(&getContext());

  // if overflow detection is enable, then rewrite to CAPI functions that
  // performs the detection, otherwise, rewrite as simple arithmetic ops
  if (enableOverflowDetection) {
    patterns
        .insert<AddOpPattern<TFHE::AddGLWEOp, TFHE::AddGLWEOp::Adaptor>,
                AddOpPattern<TFHE::AddGLWEIntOp, TFHE::AddGLWEIntOp::Adaptor>,
                MulOpPattern>(&getContext(), converter);
  } else {
    patterns.insert<mlir::concretelang::GenericOneToOneOpConversionPattern<
                        TFHE::AddGLWEIntOp, mlir::arith::AddIOp>,
                    mlir::concretelang::GenericOneToOneOpConversionPattern<
                        TFHE::AddGLWEOp, mlir::arith::AddIOp>,
                    mlir::concretelang::GenericOneToOneOpConversionPattern<
                        TFHE::MulGLWEIntOp, mlir::arith::MulIOp>>(&getContext(),
                                                                  converter);
  }

  patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::func::ReturnOp>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::scf::YieldOp>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::bufferization::AllocTensorOp, true>>(&getContext(),
                                                              converter);

  mlir::concretelang::populateWithRTTypeConverterPatterns(patterns, target,
                                                          converter);

  // Make sure that functions no longer operate on ciphertexts
  target.addDynamicallyLegalOp<mlir::func::FuncOp>(
      [&](mlir::func::FuncOp funcOp) {
        return converter.isSignatureLegal(funcOp.getFunctionType()) &&
               converter.isLegal(&funcOp.getBody());
      });
  target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
      [&](mlir::func::ConstantOp op) {
        return FunctionConstantOpConversion<SimulateTFHETypeConverter>::isLegal(
            op, converter);
      });
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
      target, converter);
  patterns.insert<mlir::concretelang::TypeConvertingReinstantiationPattern<
      mlir::func::ReturnOp>>(&getContext(), converter);

  patterns.add<FunctionConstantOpConversion<SimulateTFHETypeConverter>>(
      &getContext(), converter);

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createSimulateTFHEPass(bool enableOverflowDetection) {
  return std::make_unique<SimulateTFHEPass>(enableOverflowDetection);
}
} // namespace concretelang
} // namespace mlir

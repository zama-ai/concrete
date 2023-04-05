// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/Dialects/SCF.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
#include "concretelang/Conversion/Utils/RTOpConverter.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "concretelang/Support/Constants.h"

namespace TFHE = mlir::concretelang::TFHE;
namespace Concrete = mlir::concretelang::Concrete;
namespace Tracing = mlir::concretelang::Tracing;

namespace {
struct TFHEToConcretePass : public TFHEToConcreteBase<TFHEToConcretePass> {
  void runOnOperation() final;
};
} // namespace

using mlir::concretelang::TFHE::GLWECipherTextType;

/// TFHEToConcreteTypeConverter is a TypeConverter that transform
/// `TFHE.glwe<sk(id){dimension,1}>` to `tensor<dimension+1, i64>>`
/// `tensor<...xTFHE.glwe<sk(id){dimension,1}>>` to
/// `tensor<...xdimension+1, i64>>`
class TFHEToConcreteTypeConverter : public mlir::TypeConverter {

public:
  TFHEToConcreteTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      assert(type.getKey().isNormalized() && "keys should be normalized");
      assert(type.getKey().getNormalized().value().polySize == 1 &&
             "converter doesn't support polynomialSize > 1");
      llvm::SmallVector<int64_t, 2> shape;
      shape.push_back(type.getKey().getNormalized().value().dimension + 1);
      return mlir::RankedTensorType::get(
          shape, mlir::IntegerType::get(type.getContext(), 64));
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto glwe = type.getElementType().dyn_cast_or_null<GLWECipherTextType>();
      if (glwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::SmallVector<int64_t> newShape;
      newShape.reserve(type.getShape().size() + 1);
      newShape.append(type.getShape().begin(), type.getShape().end());
      assert(glwe.getKey().isNormalized());
      newShape.push_back(glwe.getKey().getNormalized().value().dimension + 1);
      mlir::Type r = mlir::RankedTensorType::get(
          newShape, mlir::IntegerType::get(type.getContext(), 64));
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

struct SubIntGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::SubGLWEIntOp> {

  SubIntGLWEOpPattern(mlir::MLIRContext *context,
                      mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::SubGLWEIntOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::SubGLWEIntOp subOp, TFHE::SubGLWEIntOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::Value negated = rewriter.create<Concrete::NegateLweTensorOp>(
        subOp.getLoc(), adaptor.getB().getType(), adaptor.getB());

    rewriter.replaceOpWithNewOp<Concrete::AddPlaintextLweTensorOp>(
        subOp, this->getTypeConverter()->convertType(subOp.getType()), negated,
        subOp.getA());

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
    auto bskIndex = bsOp.getKeyAttr().getIndex();

    rewriter.replaceOpWithNewOp<Concrete::BootstrapLweTensorOp>(
        bsOp, this->getTypeConverter()->convertType(resultType),
        adaptor.getCiphertext(), adaptor.getLookupTable(), inputLweDimension,
        polySize, levels, baseLog, glweDimension, bskIndex);

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
  matchAndRewrite(TFHE::WopPBSGLWEOp op, TFHE::WopPBSGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    auto bsBaseLog = adaptor.getBsk().getBaseLog();
    auto bsLevels = adaptor.getBsk().getLevels();
    auto cbsBaseLog = adaptor.getCbsBaseLog();
    auto cbsLevels = adaptor.getCbsLevels();
    auto ksBaseLog = adaptor.getKsk().getBaseLog();
    auto ksLevels = adaptor.getKsk().getLevels();
    auto pksBaseLog = adaptor.getPksk().getBaseLog();
    auto pksLevels = adaptor.getPksk().getLevels();
    auto pksInnerLweDim = adaptor.getPksk().getInnerLweDim();
    auto pksOutputPolySize = adaptor.getPksk().getOutputPolySize();
    auto crtDecomposition = adaptor.getCrtDecompositionAttr();
    auto resultType = op.getType();
    auto kskIndex = op.getKskAttr().getIndex();
    auto bskIndex = op.getBskAttr().getIndex();
    auto pkskIndex = op.getPkskAttr().getIndex();

    rewriter.replaceOpWithNewOp<Concrete::WopPBSCRTLweTensorOp>(
        op, this->getTypeConverter()->convertType(resultType),
        adaptor.getCiphertexts(), adaptor.getLookupTable(), bsLevels, bsBaseLog,
        ksLevels, ksBaseLog, pksInnerLweDim, pksOutputPolySize, pksLevels,
        pksBaseLog, cbsLevels, cbsBaseLog, crtDecomposition, kskIndex, bskIndex,
        pkskIndex);

    return mlir::success();
  }
};

struct BatchedBootstrapGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::BatchedBootstrapGLWEOp> {

  BatchedBootstrapGLWEOpPattern(mlir::MLIRContext *context,
                                mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::BatchedBootstrapGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::BatchedBootstrapGLWEOp bbsOp,
                  TFHE::BatchedBootstrapGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TFHE::GLWECipherTextType inputElementType =
        bbsOp.getCiphertexts()
            .getType()
            .cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<TFHE::GLWECipherTextType>();

    auto polySize = adaptor.getKey().getPolySize();
    auto glweDimension = adaptor.getKey().getGlweDim();
    auto levels = adaptor.getKey().getLevels();
    auto baseLog = adaptor.getKey().getBaseLog();
    auto inputLweDimension =
        inputElementType.getKey().getNormalized().value().dimension;
    auto bskIndex = adaptor.getKey().getIndex();

    rewriter.replaceOpWithNewOp<Concrete::BatchedBootstrapLweTensorOp>(
        bbsOp, this->getTypeConverter()->convertType(bbsOp.getType()),
        adaptor.getCiphertexts(), adaptor.getLookupTable(), inputLweDimension,
        polySize, levels, baseLog, glweDimension, bskIndex);

    return mlir::success();
  }
};

struct BatchedMappedBootstrapGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::BatchedMappedBootstrapGLWEOp> {

  BatchedMappedBootstrapGLWEOpPattern(mlir::MLIRContext *context,
                                      mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::BatchedMappedBootstrapGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::BatchedMappedBootstrapGLWEOp bmbsOp,
                  TFHE::BatchedMappedBootstrapGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    TFHE::GLWECipherTextType inputElementType =
        bmbsOp.getCiphertexts()
            .getType()
            .cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<TFHE::GLWECipherTextType>();

    auto polySize = adaptor.getKey().getPolySize();
    auto glweDimension = adaptor.getKey().getGlweDim();
    auto levels = adaptor.getKey().getLevels();
    auto baseLog = adaptor.getKey().getBaseLog();
    auto inputLweDimension =
        inputElementType.getKey().getNormalized().value().dimension;
    auto bskIndex = bmbsOp.getKeyAttr().getIndex();

    rewriter.replaceOpWithNewOp<Concrete::BatchedMappedBootstrapLweTensorOp>(
        bmbsOp, this->getTypeConverter()->convertType(bmbsOp.getType()),
        adaptor.getCiphertexts(), adaptor.getLookupTable(), inputLweDimension,
        polySize, levels, baseLog, glweDimension, bskIndex);

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

    TFHE::GLWECipherTextType resultType =
        ksOp.getType().cast<TFHE::GLWECipherTextType>();
    TFHE::GLWECipherTextType inputType =
        ksOp.getCiphertext().getType().cast<TFHE::GLWECipherTextType>();

    auto levels = adaptor.getKey().getLevels();
    auto baseLog = adaptor.getKey().getBaseLog();
    auto inputDim = inputType.getKey().getNormalized().value().dimension;
    auto outputDim = resultType.getKey().getNormalized().value().dimension;
    auto kskIndex = ksOp.getKeyAttr().getIndex();

    rewriter.replaceOpWithNewOp<Concrete::KeySwitchLweTensorOp>(
        ksOp, this->getTypeConverter()->convertType(resultType),
        adaptor.getCiphertext(), levels, baseLog, inputDim, outputDim,
        kskIndex);

    return mlir::success();
  }
};

struct BatchedKeySwitchGLWEOpPattern
    : public mlir::OpConversionPattern<TFHE::BatchedKeySwitchGLWEOp> {

  BatchedKeySwitchGLWEOpPattern(mlir::MLIRContext *context,
                                mlir::TypeConverter &typeConverter)
      : mlir::OpConversionPattern<TFHE::BatchedKeySwitchGLWEOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(TFHE::BatchedKeySwitchGLWEOp bksOp,
                  TFHE::BatchedKeySwitchGLWEOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {

    TFHE::GLWECipherTextType resultElementType =
        bksOp.getType()
            .cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<TFHE::GLWECipherTextType>();
    TFHE::GLWECipherTextType inputElementType =
        bksOp.getCiphertexts()
            .getType()
            .cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<TFHE::GLWECipherTextType>();

    auto levels = adaptor.getKey().getLevels();
    auto baseLog = adaptor.getKey().getBaseLog();
    auto inputDim = inputElementType.getKey().getNormalized().value().dimension;
    auto outputDim =
        resultElementType.getKey().getNormalized().value().dimension;
    auto kskIndex = adaptor.getKey().getIndex();

    rewriter.replaceOpWithNewOp<Concrete::BatchedKeySwitchLweTensorOp>(
        bksOp, this->getTypeConverter()->convertType(bksOp.getType()),
        adaptor.getCiphertexts(), levels, baseLog, inputDim, outputDim,
        kskIndex);

    return mlir::success();
  }
};

struct TracePlaintextOpPattern
    : public mlir::OpRewritePattern<Tracing::TracePlaintextOp> {
  TracePlaintextOpPattern(mlir::MLIRContext *context,
                          mlir::TypeConverter &converter,
                          mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<Tracing::TracePlaintextOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(Tracing::TracePlaintextOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto inputWidth =
        op.getPlaintext().getType().cast<mlir::IntegerType>().getWidth();
    if (inputWidth == 64) {
      op->setAttr("input_width", rewriter.getI64IntegerAttr(inputWidth));
      return mlir::success();
    }
    auto extendedInput = rewriter.create<mlir::arith::ExtUIOp>(
        op.getLoc(), rewriter.getI64Type(), op.getPlaintext());
    auto newOp = rewriter.replaceOpWithNewOp<Tracing::TracePlaintextOp>(
        op, extendedInput, op.getMsgAttr(), op.getNmsbAttr());
    newOp->setAttr("input_width", rewriter.getI64IntegerAttr(inputWidth));
    return ::mlir::success();
  }
};

template <typename ZeroOp>
struct ZeroOpPattern : public mlir::OpConversionPattern<ZeroOp> {
  ZeroOpPattern(mlir::MLIRContext *context, mlir::TypeConverter &converter)
      : mlir::OpConversionPattern<ZeroOp>(
            converter, context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(ZeroOp zeroOp, typename ZeroOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto newResultTy = this->getTypeConverter()->convertType(zeroOp.getType());

    auto generateBody = [&](mlir::OpBuilder &nestedBuilder,
                            mlir::Location nestedLoc,
                            mlir::ValueRange blockArgs) {
      // %c0 = 0 : i64
      auto cstOp = nestedBuilder.create<mlir::arith::ConstantOp>(
          nestedLoc, nestedBuilder.getI64IntegerAttr(0));
      // tensor.yield %z : !FHE.eint<p>
      nestedBuilder.create<mlir::tensor::YieldOp>(nestedLoc, cstOp.getResult());
    };
    // tensor.generate
    rewriter.replaceOpWithNewOp<mlir::tensor::GenerateOp>(
        zeroOp, newResultTy, mlir::ValueRange{}, generateBody);

    return ::mlir::success();
  };
};

/// Pattern that rewrites the ExtractSlice operation, taking into account the
/// additional LWE dimension introduced during type conversion
struct ExtractSliceOpPattern
    : public mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp> {
  ExtractSliceOpPattern(mlir::MLIRContext *context,
                        mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<mlir::tensor::ExtractSliceOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp extractSliceOp,
                  mlir::tensor::ExtractSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // is not a tensor of GLWEs that need to be extended with the LWE dimension
    if (this->getTypeConverter()->isLegal(extractSliceOp.getType())) {
      return mlir::failure();
    }
    auto resultTy = extractSliceOp.getResult().getType();
    auto newResultTy = this->getTypeConverter()
                           ->convertType(resultTy)
                           .cast<mlir::RankedTensorType>();

    // add 0 to the static_offsets
    mlir::SmallVector<int64_t> staticOffsets;
    staticOffsets.append(adaptor.getStaticOffsets().begin(),
                         adaptor.getStaticOffsets().end());
    staticOffsets.push_back(0);

    // add the lweSize to the sizes
    mlir::SmallVector<int64_t> staticSizes;
    staticSizes.append(adaptor.getStaticSizes().begin(),
                       adaptor.getStaticSizes().end());
    staticSizes.push_back(newResultTy.getDimSize(newResultTy.getRank() - 1));

    // add 1 to the strides
    mlir::SmallVector<int64_t> staticStrides;
    staticStrides.append(adaptor.getStaticStrides().begin(),
                         adaptor.getStaticStrides().end());
    staticStrides.push_back(1);

    // replace tensor.extract_slice to the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        extractSliceOp, newResultTy, adaptor.getSource(), adaptor.getOffsets(),
        adaptor.getSizes(), adaptor.getStrides(),
        rewriter.getDenseI64ArrayAttr(staticOffsets),
        rewriter.getDenseI64ArrayAttr(staticSizes),
        rewriter.getDenseI64ArrayAttr(staticStrides));

    return ::mlir::success();
  };
};

/// Pattern that rewrites the Extract operation, taking into account the
/// additional LWE dimension introduced during type conversion
struct ExtractOpPattern
    : public mlir::OpConversionPattern<mlir::tensor::ExtractOp> {
  ExtractOpPattern(::mlir::MLIRContext *context,
                   mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<mlir::tensor::ExtractOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractOp extractOp,
                  mlir::tensor::ExtractOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // is not a tensor of GLWEs that need to be extended with the LWE dimension
    if (this->getTypeConverter()->isLegal(extractOp.getType())) {
      return mlir::failure();
    }

    auto newResultType = this->getTypeConverter()
                             ->convertType(extractOp.getType())
                             .cast<mlir::RankedTensorType>();
    auto tensorRank =
        adaptor.getTensor().getType().cast<mlir::RankedTensorType>().getRank();

    // [min..., 0] for static_offsets ()
    mlir::SmallVector<int64_t> staticOffsets(
        tensorRank, std::numeric_limits<int64_t>::min());
    staticOffsets[staticOffsets.size() - 1] = 0;

    // [1..., lweDimension+1] for static_sizes or
    // [1..., nbBlock, lweDimension+1]
    mlir::SmallVector<int64_t> staticSizes(tensorRank, 1);
    staticSizes[staticSizes.size() - 1] =
        newResultType.getDimSize(newResultType.getRank() - 1);

    // [1...] for static_strides
    mlir::SmallVector<int64_t> staticStrides(tensorRank, 1);

    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        extractOp, newResultType, adaptor.getTensor(), adaptor.getIndices(),
        mlir::SmallVector<mlir::Value>{}, mlir::SmallVector<mlir::Value>{},
        rewriter.getDenseI64ArrayAttr(staticOffsets),
        rewriter.getDenseI64ArrayAttr(staticSizes),
        rewriter.getDenseI64ArrayAttr(staticStrides));

    return ::mlir::success();
  };
};

/// Pattern that rewrites the InsertSlice-like operation, taking into
/// account the additional LWE dimension introduced during type
/// conversion
template <typename OpTy>
struct InsertSliceOpPattern : public mlir::OpConversionPattern<OpTy> {
  InsertSliceOpPattern(mlir::MLIRContext *context,
                       mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<OpTy>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(OpTy insertSliceOp, typename OpTy::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    mlir::RankedTensorType newDestTy = ((mlir::Type)adaptor.getDest().getType())
                                           .cast<mlir::RankedTensorType>();

    // add 0 to offsets
    mlir::SmallVector<mlir::OpFoldResult> offsets = getMixedValues(
        adaptor.getStaticOffsets(), adaptor.getOffsets(), rewriter);
    offsets.push_back(rewriter.getI64IntegerAttr(0));

    // add lweDimension+1 to sizes
    mlir::SmallVector<mlir::OpFoldResult> sizes =
        getMixedValues(adaptor.getStaticSizes(), adaptor.getSizes(), rewriter);
    sizes.push_back(rewriter.getI64IntegerAttr(
        newDestTy.getDimSize(newDestTy.getRank() - 1)));

    // add 1 to the strides
    mlir::SmallVector<mlir::OpFoldResult> strides = getMixedValues(
        adaptor.getStaticStrides(), adaptor.getStrides(), rewriter);
    strides.push_back(rewriter.getI64IntegerAttr(1));

    // replace insert slice-like operation with the new one
    rewriter.replaceOpWithNewOp<OpTy>(insertSliceOp, adaptor.getSource(),
                                      adaptor.getDest(), offsets, sizes,
                                      strides);

    return ::mlir::success();
  };
};

/// Pattern that rewrites the Insert operation, taking into account the
/// additional LWE dimension introduced during type conversion
struct InsertOpPattern
    : public mlir::OpConversionPattern<mlir::tensor::InsertOp> {
  InsertOpPattern(mlir::MLIRContext *context,
                  mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<mlir::tensor::InsertOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertOp insertOp,
                  mlir::tensor::InsertOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // is not a tensor of GLWEs that need to be extended with the LWE dimension
    if (this->getTypeConverter()->isLegal(insertOp.getType())) {
      return mlir::failure();
    }

    mlir::RankedTensorType newResultTy =
        this->getTypeConverter()
            ->convertType(insertOp.getResult().getType())
            .cast<mlir::RankedTensorType>();

    // add zeros to static offsets
    mlir::SmallVector<mlir::OpFoldResult> offsets;
    offsets.append(adaptor.getIndices().begin(), adaptor.getIndices().end());
    offsets.push_back(rewriter.getIndexAttr(0));

    // Inserting a smaller tensor into a (potentially) bigger one. Set
    // dimensions for all leading dimensions of the target tensor not
    // present in the source to 1.
    mlir::SmallVector<mlir::OpFoldResult> sizes(adaptor.getIndices().size(),
                                                rewriter.getI64IntegerAttr(1));

    // Add size for the bufferized source element
    sizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // Set stride of all dimensions to 1
    mlir::SmallVector<mlir::OpFoldResult> strides(
        newResultTy.getRank(), rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        insertOp, adaptor.getScalar(), adaptor.getDest(), offsets, sizes,
        strides);

    return ::mlir::success();
  };
};

/// FromElementsOpPatterns transform each tensor.from_elements that operates on
/// TFHE.glwe
///
/// refs: check_tests/Conversion/TFHEToConcrete/tensor_from_elements.mlir
struct FromElementsOpPattern
    : public mlir::OpConversionPattern<mlir::tensor::FromElementsOp> {
  FromElementsOpPattern(mlir::MLIRContext *context,
                        mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<mlir::tensor::FromElementsOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::FromElementsOp fromElementsOp,
                  mlir::tensor::FromElementsOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {

    // is not a tensor of GLWEs that need to be extended with the LWE dimension
    if (this->getTypeConverter()->isLegal(fromElementsOp.getType())) {
      return mlir::failure();
    }

    auto converter = this->getTypeConverter();

    auto resultTy = fromElementsOp.getResult().getType();
    if (converter->isLegal(resultTy)) {
      return mlir::failure();
    }
    auto oldTensorResultTy = resultTy.cast<mlir::RankedTensorType>();
    auto oldRank = oldTensorResultTy.getRank();

    auto newTensorResultTy =
        converter->convertType(resultTy).cast<mlir::RankedTensorType>();
    auto newRank = newTensorResultTy.getRank();
    auto newShape = newTensorResultTy.getShape();

    mlir::Value tensor = rewriter.create<mlir::bufferization::AllocTensorOp>(
        fromElementsOp.getLoc(), newTensorResultTy, mlir::ValueRange{});

    // sizes are [1, ..., 1, diffShape...]
    llvm::SmallVector<mlir::OpFoldResult> sizes(oldRank,
                                                rewriter.getI64IntegerAttr(1));
    for (auto i = newRank - oldRank; i > 0; i--) {
      sizes.push_back(rewriter.getI64IntegerAttr(*(newShape.end() - i)));
    }

    // strides are [1, ..., 1]
    llvm::SmallVector<mlir::OpFoldResult> oneStrides(
        newShape.size(), rewriter.getI64IntegerAttr(1));

    // start with offets [0, ..., 0]
    llvm::SmallVector<int64_t> currentOffsets(newRank, 0);

    // for each elements insert_slice with right offet
    for (auto elt : llvm::enumerate(adaptor.getElements())) {
      // Just create offsets as attributes
      llvm::SmallVector<mlir::OpFoldResult, 4> offsets;
      offsets.reserve(currentOffsets.size());
      std::transform(currentOffsets.begin(), currentOffsets.end(),
                     std::back_inserter(offsets),
                     [&](auto v) { return rewriter.getI64IntegerAttr(v); });
      mlir::tensor::InsertSliceOp insOp =
          rewriter.create<mlir::tensor::InsertSliceOp>(
              fromElementsOp.getLoc(),
              /* src: */ elt.value(),
              /* dst: */ tensor,
              /* offs: */ offsets,
              /* sizes: */ sizes,
              /* strides: */ oneStrides);

      tensor = insOp.getResult();

      // Increment the offsets
      for (auto i = newRank - 2; i >= 0; i--) {
        if (currentOffsets[i] == newShape[i] - 1) {
          currentOffsets[i] = 0;
          continue;
        }
        currentOffsets[i]++;
        break;
      }
    }

    rewriter.replaceOp(fromElementsOp, tensor);
    return ::mlir::success();
  };
};

// This template rewrite pattern transforms any instance of
// `ShapeOp` operators that operates on tensor of lwe ciphertext by adding
// the lwe size as a size of the tensor result and by adding a trivial
// reassociation at the end of the reassociations map.
//
// Example:
//
// ```mlir
// %0 = "ShapeOp" %arg0 [reassocations...]
//        : tensor<...x!TFHE.glwe<sk(id){dimension,1}>> into
//          tensor<...x!TFHE.glwe<sk(id){dimension,1}>>
// ```
//
// becomes:
//
// ```mlir
// %0 = "ShapeOp" %arg0 [reassociations..., [inRank or outRank]]
//        : tensor<...xdimension+1xi64> into
//          tensor<...xdimension+1xi64>
// ```
template <typename ShapeOp, typename ShapeOpAdaptor, typename VecTy,
          bool inRank>
struct TensorShapeOpPattern : public mlir::OpConversionPattern<ShapeOp> {
  TensorShapeOpPattern(mlir::MLIRContext *context,
                       mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<ShapeOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(ShapeOp shapeOp, ShapeOpAdaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // is not a tensor of GLWEs that need to be extended with the LWE dimension
    if (this->getTypeConverter()->isLegal(shapeOp.getType())) {
      return mlir::failure();
    }

    auto newResultTy =
        ((mlir::Type)this->getTypeConverter()->convertType(shapeOp.getType()))
            .cast<VecTy>();

    auto reassocTy =
        ((mlir::Type)this->getTypeConverter()->convertType(
             (inRank ? shapeOp.getSrc() : shapeOp.getResult()).getType()))
            .cast<VecTy>();

    auto oldReassocs = shapeOp.getReassociationIndices();
    mlir::SmallVector<mlir::ReassociationIndices> newReassocs;
    newReassocs.append(oldReassocs.begin(), oldReassocs.end());

    // add [rank] to reassociations
    {
      mlir::ReassociationIndices lweAssoc;
      lweAssoc.push_back(reassocTy.getRank() - 1);
      newReassocs.push_back(lweAssoc);
    }

    rewriter.replaceOpWithNewOp<ShapeOp>(shapeOp, newResultTy, adaptor.getSrc(),
                                         newReassocs);

    return ::mlir::success();
  };
};

/// Add the instantiated TensorShapeOpPattern rewrite pattern with the
/// `ShapeOp` to the patterns set and populate the conversion target.
template <typename ShapeOp, typename ShapeOpAdaptor, typename VecTy,
          bool inRank>
void insertTensorShapeOpPattern(mlir::MLIRContext &context,
                                mlir::TypeConverter &converter,
                                mlir::RewritePatternSet &patterns,
                                mlir::ConversionTarget &target) {
  patterns.insert<TensorShapeOpPattern<ShapeOp, ShapeOpAdaptor, VecTy, inRank>>(
      &context, converter);
  target.addDynamicallyLegalOp<ShapeOp>([&](mlir::Operation *op) {
    return converter.isLegal(op->getResultTypes()) &&
           converter.isLegal(op->getOperandTypes());
  });
}

// The pass is supposed to endup with no TFHE.glwe type. Tensors should be
// extended with an additional dimension at the end, and some patterns in this
// pass are fully dedicated to rewrite tensor ops with this additional dimension
// in mind
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

  target.addDynamicallyLegalOp<mlir::scf::ForallOp>(
      [&](mlir::scf::ForallOp op) {
        return (
            converter.isLegal(op->getOperandTypes()) &&
            converter.isLegal(op->getResultTypes()) &&
            converter.isLegal(op->getRegion(0).front().getArgumentTypes()) &&
            converter.isLegal(op.getOutputs().getTypes()));
      });

  target.addDynamicallyLegalOp<mlir::scf::InParallelOp>(
      [&](mlir::scf::InParallelOp op) {
        return converter.isLegal(&op.getBodyRegion());
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
  //   populateWithGeneratedTFHEToConcrete(patterns);

  // Generic patterns
  patterns.insert<
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::AddGLWEOp,
          mlir::concretelang::Concrete::AddLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::AddGLWEIntOp,
          mlir::concretelang::Concrete::AddPlaintextLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::MulGLWEIntOp,
          mlir::concretelang::Concrete::MulCleartextLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::NegGLWEOp,
          mlir::concretelang::Concrete::NegateLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::EncodeExpandLutForBootstrapOp,
          mlir::concretelang::Concrete::EncodeExpandLutForBootstrapTensorOp,
          true>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::EncodeLutForCrtWopPBSOp,
          mlir::concretelang::Concrete::EncodeLutForCrtWopPBSTensorOp, true>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::EncodePlaintextWithCrtOp,
          mlir::concretelang::Concrete::EncodePlaintextWithCrtTensorOp, true>,

      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::ABatchedAddGLWEIntOp,
          mlir::concretelang::Concrete::BatchedAddPlaintextLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::ABatchedAddGLWEIntCstOp,
          mlir::concretelang::Concrete::BatchedAddPlaintextCstLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::ABatchedAddGLWEOp,
          mlir::concretelang::Concrete::BatchedAddLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::BatchedMulGLWEIntOp,
          mlir::concretelang::Concrete::BatchedMulCleartextLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::BatchedMulGLWEIntCstOp,
          mlir::concretelang::Concrete::BatchedMulCleartextCstLweTensorOp>,
      mlir::concretelang::GenericOneToOneOpConversionPattern<
          mlir::concretelang::TFHE::BatchedNegGLWEOp,
          mlir::concretelang::Concrete::BatchedNegateLweTensorOp>

      >(&getContext(), converter);
  // pattern of remaining TFHE ops

  patterns.insert<ZeroOpPattern<mlir::concretelang::TFHE::ZeroGLWEOp>,
                  ZeroOpPattern<mlir::concretelang::TFHE::ZeroTensorGLWEOp>,
                  SubIntGLWEOpPattern, BootstrapGLWEOpPattern,
                  BatchedBootstrapGLWEOpPattern,
                  BatchedMappedBootstrapGLWEOpPattern, KeySwitchGLWEOpPattern,
                  BatchedKeySwitchGLWEOpPattern, WopPBSGLWEOpPattern>(
      &getContext(), converter);

  // Add patterns to rewrite tensor operators that works on tensors of TFHE GLWE
  // types
  patterns.insert<ExtractSliceOpPattern, ExtractOpPattern,
                  InsertSliceOpPattern<mlir::tensor::InsertSliceOp>,
                  InsertSliceOpPattern<mlir::tensor::ParallelInsertSliceOp>,
                  InsertOpPattern, FromElementsOpPattern>(&getContext(),
                                                          converter);
  // Add patterns to rewrite some of tensor ops that were introduced by the
  // linalg bufferization of encrypted tensor
  insertTensorShapeOpPattern<mlir::tensor::ExpandShapeOp,
                             mlir::tensor::ExpandShapeOp::Adaptor,
                             mlir::TensorType, false>(getContext(), converter,
                                                      patterns, target);
  insertTensorShapeOpPattern<mlir::tensor::CollapseShapeOp,
                             mlir::tensor::CollapseShapeOp::Adaptor,
                             mlir::TensorType, true>(getContext(), converter,
                                                     patterns, target);
  // legalize ops only if operand and result types are legal
  target.addDynamicallyLegalOp<
      mlir::tensor::YieldOp, mlir::scf::YieldOp, mlir::tensor::GenerateOp,
      mlir::tensor::ExtractSliceOp, mlir::tensor::ExtractOp,
      mlir::tensor::InsertSliceOp, mlir::tensor::ParallelInsertSliceOp,
      mlir::tensor::ExpandShapeOp, mlir::tensor::CollapseShapeOp,
      mlir::tensor::EmptyOp, mlir::bufferization::AllocTensorOp>(
      [&](mlir::Operation *op) {
        return converter.isLegal(op->getResultTypes()) &&
               converter.isLegal(op->getOperandTypes());
      });

  // rewrite scf for loops if working on illegal types
  patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::scf::ForOp>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::scf::ForallOp>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::scf::InParallelOp>>(&getContext(), converter);

  mlir::concretelang::addDynamicallyLegalTypeOp<mlir::func::ReturnOp>(
      target, converter);
  mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
      patterns, converter);

  // Conversion of Tracing dialect
  patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
      Tracing::TraceCiphertextOp, true>>(&getContext(), converter);
  mlir::concretelang::addDynamicallyLegalTypeOp<Tracing::TraceCiphertextOp>(
      target, converter);
  patterns.add<TracePlaintextOpPattern>(&getContext(), converter);
  target.addLegalOp<mlir::arith::ExtUIOp>();
  target.addDynamicallyLegalOp<Tracing::TracePlaintextOp>(
      [&](Tracing::TracePlaintextOp op) {
        return (
            op.getPlaintext().getType().cast<mlir::IntegerType>().getWidth() ==
            64);
      });

  patterns.add<mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::func::ReturnOp>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::scf::YieldOp>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::bufferization::AllocTensorOp, true>,
               mlir::concretelang::TypeConvertingReinstantiationPattern<
                   mlir::tensor::EmptyOp>>(&getContext(), converter);

  mlir::concretelang::populateWithRTTypeConverterPatterns(patterns, target,
                                                          converter);

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

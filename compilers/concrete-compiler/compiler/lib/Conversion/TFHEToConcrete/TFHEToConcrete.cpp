// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <iostream>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/FuncConstOpConversion.h"
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
/// `TFHE.glwe<{dimension,1,bits}{p}>` to `tensor<dimension+1, i64>>`
/// `tensor<...xTFHE.glwe<{dimension,1,bits}{p}>>` to
/// `tensor<...xdimension+1, i64>>`
class TFHEToConcreteTypeConverter : public mlir::TypeConverter {

public:
  TFHEToConcreteTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](GLWECipherTextType type) {
      assert(type.getPolynomialSize() <= 1 &&
             "converter doesn't support polynomialSize > 1");
      assert(type.getDimension() != -1);
      llvm::SmallVector<int64_t, 2> shape;
      shape.push_back(type.getDimension() + 1);
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
      assert(glwe.getDimension() != -1);
      newShape.push_back(glwe.getDimension() + 1);
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
        subOp.getLoc(), adaptor.b().getType(), adaptor.b());

    rewriter.replaceOpWithNewOp<Concrete::AddPlaintextLweTensorOp>(
        subOp, this->getTypeConverter()->convertType(subOp.getType()), negated,
        subOp.a());

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
        bsOp.ciphertext().getType().cast<TFHE::GLWECipherTextType>();

    rewriter.replaceOpWithNewOp<Concrete::BootstrapLweTensorOp>(
        bsOp, this->getTypeConverter()->convertType(resultType),
        adaptor.ciphertext(), adaptor.lookup_table(), inputType.getDimension(),
        adaptor.polySize(), adaptor.level(), adaptor.baseLog(),
        adaptor.glweDimension(), resultType.getP());

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
        ksOp.ciphertext().getType().cast<TFHE::GLWECipherTextType>();

    rewriter.replaceOpWithNewOp<Concrete::KeySwitchLweTensorOp>(
        ksOp, this->getTypeConverter()->convertType(resultType),
        adaptor.ciphertext(), adaptor.level(), adaptor.baseLog(),
        inputType.getDimension(), resultType.getDimension());

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
        op.plaintext().getType().cast<mlir::IntegerType>().getWidth();
    if (inputWidth == 64) {
      op->setAttr("input_width", rewriter.getI64IntegerAttr(inputWidth));
      return mlir::success();
    }
    auto extendedInput = rewriter.create<mlir::arith::ExtUIOp>(
        op.getLoc(), rewriter.getI64Type(), op.plaintext());
    auto newOp = rewriter.replaceOpWithNewOp<Tracing::TracePlaintextOp>(
        op, extendedInput, op.msgAttr(), op.nmsbAttr());
    newOp->setAttr("input_width", rewriter.getI64IntegerAttr(inputWidth));
    return ::mlir::success();
  }
};

template <typename ZeroOp>
struct ZeroOpPattern : public mlir::OpRewritePattern<ZeroOp> {
  ZeroOpPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<ZeroOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(ZeroOp zeroOp,
                  mlir::PatternRewriter &rewriter) const override {
    TFHEToConcreteTypeConverter converter;
    auto newResultTy = converter.convertType(zeroOp.getType());

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
    auto resultTy = extractSliceOp.result().getType();
    auto newResultTy = this->getTypeConverter()
                           ->convertType(resultTy)
                           .cast<mlir::RankedTensorType>();

    // add 0 to the static_offsets
    mlir::SmallVector<mlir::Attribute> staticOffsets;
    staticOffsets.append(adaptor.static_offsets().begin(),
                         adaptor.static_offsets().end());
    staticOffsets.push_back(rewriter.getI64IntegerAttr(0));

    // add the lweSize to the sizes
    mlir::SmallVector<mlir::Attribute> staticSizes;
    staticSizes.append(adaptor.static_sizes().begin(),
                       adaptor.static_sizes().end());
    staticSizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // add 1 to the strides
    mlir::SmallVector<mlir::Attribute> staticStrides;
    staticStrides.append(adaptor.static_strides().begin(),
                         adaptor.static_strides().end());
    staticStrides.push_back(rewriter.getI64IntegerAttr(1));

    // replace tensor.extract_slice to the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        extractSliceOp, newResultTy, adaptor.source(), adaptor.offsets(),
        adaptor.sizes(), adaptor.strides(),
        rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

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
        adaptor.tensor().getType().cast<mlir::RankedTensorType>().getRank();

    // [min..., 0] for static_offsets ()
    mlir::SmallVector<mlir::Attribute> staticOffsets(
        tensorRank,
        rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
    staticOffsets[staticOffsets.size() - 1] = rewriter.getI64IntegerAttr(0);

    // [1..., lweDimension+1] for static_sizes or
    // [1..., nbBlock, lweDimension+1]
    mlir::SmallVector<mlir::Attribute> staticSizes(
        tensorRank, rewriter.getI64IntegerAttr(1));
    staticSizes[staticSizes.size() - 1] = rewriter.getI64IntegerAttr(
        newResultType.getDimSize(newResultType.getRank() - 1));

    // [1...] for static_strides
    mlir::SmallVector<mlir::Attribute> staticStrides(
        tensorRank, rewriter.getI64IntegerAttr(1));

    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        extractOp, newResultType, adaptor.tensor(), adaptor.indices(),
        mlir::SmallVector<mlir::Value>{}, mlir::SmallVector<mlir::Value>{},
        rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

    return ::mlir::success();
  };
};

/// Pattern that rewrites the InsertSlice operation, taking into account the
/// additional LWE dimension introduced during type conversion
struct InsertSliceOpPattern
    : public mlir::OpConversionPattern<mlir::tensor::InsertSliceOp> {
  InsertSliceOpPattern(mlir::MLIRContext *context,
                       mlir::TypeConverter &typeConverter)
      : ::mlir::OpConversionPattern<mlir::tensor::InsertSliceOp>(
            typeConverter, context,
            mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertSliceOp insertSliceOp,
                  mlir::tensor::InsertSliceOp::Adaptor adaptor,
                  ::mlir::ConversionPatternRewriter &rewriter) const override {
    // is not a tensor of GLWEs that need to be extended with the LWE dimension
    if (this->getTypeConverter()->isLegal(insertSliceOp.getType())) {
      return mlir::failure();
    }

    auto newResultTy = this->getTypeConverter()
                           ->convertType(insertSliceOp.result().getType())
                           .cast<mlir::RankedTensorType>();

    // add 0 to static_offsets
    mlir::SmallVector<mlir::Attribute> staticOffsets;
    staticOffsets.append(adaptor.static_offsets().begin(),
                         adaptor.static_offsets().end());
    staticOffsets.push_back(rewriter.getI64IntegerAttr(0));

    // add lweDimension+1 to static_sizes
    mlir::SmallVector<mlir::Attribute> staticSizes;
    staticSizes.append(adaptor.static_sizes().begin(),
                       adaptor.static_sizes().end());
    staticSizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // add 1 to the strides
    mlir::SmallVector<mlir::Attribute> staticStrides;
    staticStrides.append(adaptor.static_strides().begin(),
                         adaptor.static_strides().end());
    staticStrides.push_back(rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        insertSliceOp, newResultTy, adaptor.source(), adaptor.dest(),
        adaptor.offsets(), adaptor.sizes(), adaptor.strides(),
        rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

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
            ->convertType(insertOp.result().getType())
            .cast<mlir::RankedTensorType>();

    // add zeros to static_offsets
    mlir::SmallVector<mlir::OpFoldResult> offsets;
    offsets.append(adaptor.indices().begin(), adaptor.indices().end());
    offsets.push_back(rewriter.getIndexAttr(0));

    // Inserting a smaller tensor into a (potentially) bigger one. Set
    // dimensions for all leading dimensions of the target tensor not
    // present in the source to 1.
    mlir::SmallVector<mlir::OpFoldResult> sizes(adaptor.indices().size(),
                                                rewriter.getI64IntegerAttr(1));

    // Add size for the bufferized source element
    sizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // Set stride of all dimensions to 1
    mlir::SmallVector<mlir::OpFoldResult> strides(
        newResultTy.getRank(), rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        insertOp, adaptor.scalar(), adaptor.dest(), offsets, sizes, strides);

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

    auto resultTy = fromElementsOp.result().getType();
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
    for (auto elt : llvm::enumerate(adaptor.elements())) {
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
//        : tensor<...x!TFHE.glwe<{dimension,1,bits}{p}>> into
//          tensor<...x!TFHE.glwe<{dimension,1,bits}{p}>>
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
             (inRank ? shapeOp.src() : shapeOp.result()).getType()))
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

    rewriter.replaceOpWithNewOp<ShapeOp>(shapeOp, newResultTy, adaptor.src(),
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
          mlir::concretelang::TFHE::WopPBSGLWEOp,
          mlir::concretelang::Concrete::WopPBSCRTLweTensorOp, true>>(
      &getContext(), converter);
  // pattern of remaining TFHE ops
  patterns.insert<ZeroOpPattern<mlir::concretelang::TFHE::ZeroGLWEOp>,
                  ZeroOpPattern<mlir::concretelang::TFHE::ZeroTensorGLWEOp>>(
      &getContext());
  patterns.insert<SubIntGLWEOpPattern, BootstrapGLWEOpPattern,
                  KeySwitchGLWEOpPattern>(&getContext(), converter);

  // Add patterns to rewrite tensor operators that works on tensors of TFHE GLWE
  // types
  patterns.insert<ExtractSliceOpPattern, ExtractOpPattern, InsertSliceOpPattern,
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
      mlir::tensor::InsertSliceOp, mlir::tensor::ExpandShapeOp,
      mlir::tensor::CollapseShapeOp, mlir::bufferization::AllocTensorOp>(
      [&](mlir::Operation *op) {
        return converter.isLegal(op->getResultTypes()) &&
               converter.isLegal(op->getOperandTypes());
      });

  // rewrite scf for loops if working on illegal types
  patterns.add<RegionOpTypeConverterPattern<mlir::scf::ForOp,
                                            TFHEToConcreteTypeConverter>>(
      &getContext(), converter);
  target.addDynamicallyLegalOp<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
    return converter.isLegal(forOp.getInitArgs().getTypes()) &&
           converter.isLegal(forOp.getResults().getTypes());
  });

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
        return (op.plaintext().getType().cast<mlir::IntegerType>().getWidth() ==
                64);
      });

  // Conversion of RT Dialect Ops
  patterns.add<
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::func::ReturnOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::scf::YieldOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::bufferization::AllocTensorOp, true>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::MakeReadyFutureOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::AwaitFutureOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::CreateAsyncTaskOp, true>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::BuildReturnPtrPlaceholderOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::DerefWorkFunctionArgumentPtrPlaceholderOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::DerefReturnPtrPlaceholderOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
          mlir::concretelang::RT::WorkFunctionReturnOp>,
      mlir::concretelang::TypeConvertingReinstantiationPattern<
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

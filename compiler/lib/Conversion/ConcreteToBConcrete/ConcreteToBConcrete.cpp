// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <iostream>

#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Utils/RegionOpTypeConverterPattern.h"
#include "concretelang/Conversion/Utils/TensorOpTypeConversion.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"

namespace {
struct ConcreteToBConcretePass
    : public ConcreteToBConcreteBase<ConcreteToBConcretePass> {
  void runOnOperation() final;
};
} // namespace

/// ConcreteToBConcreteTypeConverter is a TypeConverter that transform
/// `Concrete.lwe_ciphertext<dimension,p>` to `tensor<dimension+1, i64>>`
/// `tensor<...xConcrete.lwe_ciphertext<dimension,p>>` to
/// `tensor<...xdimension+1, i64>>`
class ConcreteToBConcreteTypeConverter : public mlir::TypeConverter {

public:
  ConcreteToBConcreteTypeConverter() {
    addConversion([](mlir::Type type) { return type; });
    addConversion([&](mlir::concretelang::Concrete::LweCiphertextType type) {
      return mlir::RankedTensorType::get(
          {type.getDimension() + 1},
          mlir::IntegerType::get(type.getContext(), 64));
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto lwe = type.getElementType()
                     .dyn_cast_or_null<
                         mlir::concretelang::Concrete::LweCiphertextType>();
      if (lwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::SmallVector<int64_t> newShape;
      newShape.reserve(type.getShape().size() + 1);
      newShape.append(type.getShape().begin(), type.getShape().end());
      newShape.push_back(lwe.getDimension() + 1);
      mlir::Type r = mlir::RankedTensorType::get(
          newShape, mlir::IntegerType::get(type.getContext(), 64));
      return r;
    });
    addConversion([&](mlir::MemRefType type) {
      auto lwe = type.getElementType()
                     .dyn_cast_or_null<
                         mlir::concretelang::Concrete::LweCiphertextType>();
      if (lwe == nullptr) {
        return (mlir::Type)(type);
      }
      mlir::SmallVector<int64_t> newShape;
      newShape.reserve(type.getShape().size() + 1);
      newShape.append(type.getShape().begin(), type.getShape().end());
      newShape.push_back(lwe.getDimension() + 1);
      mlir::Type r = mlir::MemRefType::get(
          newShape, mlir::IntegerType::get(type.getContext(), 64));
      return r;
    });
  }
};

// This rewrite pattern transforms any instance of `Concrete.zero_tensor`
// operators.
//
// Example:
//
// ```mlir
// %0 = "Concrete.zero_tensor" () :
// tensor<...x!Concrete.lwe_ciphertext<lweDim,p>>
// ```
//
// becomes:
//
// ```mlir
// %0 = tensor.generate {
//   ^bb0(... : index):
//     %c0 = arith.constant 0 : i64
//     tensor.yield %z
// }: tensor<...xlweDim+1xi64>
// i64>
// ```
template <typename ZeroOp>
struct ZeroOpPattern : public mlir::OpRewritePattern<ZeroOp> {
  ZeroOpPattern(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ZeroOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(ZeroOp zeroOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = zeroOp.getType();
    auto newResultTy = converter.convertType(resultTy);

    auto generateBody = [&](mlir::OpBuilder &nestedBuilder,
                            mlir::Location nestedLoc,
                            mlir::ValueRange blockArgs) {
      // %c0 = 0 : i64
      auto cstOp = nestedBuilder.create<mlir::arith::ConstantOp>(
          nestedLoc, nestedBuilder.getI64IntegerAttr(1));
      // tensor.yield %z : !FHE.eint<p>
      nestedBuilder.create<mlir::tensor::YieldOp>(nestedLoc, cstOp.getResult());
    };
    // tensor.generate
    rewriter.replaceOpWithNewOp<mlir::tensor::GenerateOp>(
        zeroOp, newResultTy, mlir::ValueRange{}, generateBody);

    return ::mlir::success();
  };
};

// This template rewrite pattern transforms any instance of
// `ConcreteOp` to an instance of `BConcreteOp`.
//
// Example:
//
//   %0 = "ConcreteOp"(%arg0, ...) :
//     (!Concrete.lwe_ciphertext<lwe_dimension, p>, ...) ->
//     (!Concrete.lwe_ciphertext<lwe_dimension, p>)
//
// becomes:
//
//   %0 = linalg.init_tensor [dimension+1] : tensor<dimension+1, i64>
//   "BConcreteOp"(%0, %arg0, ...) : (tensor<dimension+1, i64>>,
//      tensor<dimension+1, i64>>, ..., ) -> ()
//
// A reference to the preallocated output is always passed as the first
// argument.
template <typename ConcreteOp, typename BConcreteOp>
struct LowToBConcrete : public mlir::OpRewritePattern<ConcreteOp> {
  LowToBConcrete(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ConcreteOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(ConcreteOp concreteOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    mlir::concretelang::Concrete::LweCiphertextType resultTy =
        ((mlir::Type)concreteOp->getResult(0).getType())
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    // %0 = linalg.init_tensor [dimension+1] : tensor<dimension+1, i64>
    mlir::Value init = rewriter.replaceOpWithNewOp<mlir::linalg::InitTensorOp>(
        concreteOp, newResultTy.getShape(), newResultTy.getElementType());

    // "BConcreteOp"(%0, %arg0, ...) : (tensor<dimension+1, i64>>,
    //   tensor<dimension+1, i64>>, ..., ) -> ()
    mlir::SmallVector<mlir::Value, 3> newOperands{init};

    newOperands.append(concreteOp.getOperation()->getOperands().begin(),
                       concreteOp.getOperation()->getOperands().end());

    llvm::ArrayRef<::mlir::NamedAttribute> attributes =
        concreteOp.getOperation()->getAttrs();

    rewriter.create<BConcreteOp>(concreteOp.getLoc(),
                                 mlir::SmallVector<mlir::Type>{}, newOperands,
                                 attributes);

    return ::mlir::success();
  };
};

// This rewrite pattern transforms any instance of
// `tensor.extract_slice` operators that operates on tensor of lwe ciphertext.
//
// Example:
//
// ```mlir
// %0 = tensor.extract_slice %arg0
//   [offsets...] [sizes...] [strides...]
//   : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> to
//     tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
// ```
//
// becomes:
//
// ```mlir
// %0 = tensor.extract_slice %arg0
//   [offsets..., 0] [sizes..., lweDimension+1] [strides..., 1]
//   : tensor<...xlweDimension+1,i64> to
//     tensor<...xlweDimension+1,i64>
// ```
struct ExtractSliceOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp> {
  ExtractSliceOpPattern(::mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::ExtractSliceOp>(context,
                                                               benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractSliceOp extractSliceOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = extractSliceOp.result().getType();
    auto resultEltTy =
        resultTy.cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto newResultTy = converter.convertType(resultTy);

    // add 0 to the static_offsets
    mlir::SmallVector<mlir::Attribute> staticOffsets;
    staticOffsets.append(extractSliceOp.static_offsets().begin(),
                         extractSliceOp.static_offsets().end());
    staticOffsets.push_back(rewriter.getI64IntegerAttr(0));

    // add the lweSize to the sizes
    mlir::SmallVector<mlir::Attribute> staticSizes;
    staticSizes.append(extractSliceOp.static_sizes().begin(),
                       extractSliceOp.static_sizes().end());
    staticSizes.push_back(
        rewriter.getI64IntegerAttr(resultEltTy.getDimension() + 1));

    // add 1 to the strides
    mlir::SmallVector<mlir::Attribute> staticStrides;
    staticStrides.append(extractSliceOp.static_strides().begin(),
                         extractSliceOp.static_strides().end());
    staticStrides.push_back(rewriter.getI64IntegerAttr(1));

    // replace tensor.extract_slice to the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
        extractSliceOp, newResultTy, extractSliceOp.source(),
        extractSliceOp.offsets(), extractSliceOp.sizes(),
        extractSliceOp.strides(), rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

    return ::mlir::success();
  };
};

// This rewrite pattern transforms any instance of
// `tensor.extract` operators that operates on tensor of lwe ciphertext.
//
// Example:
//
// ```mlir
// %0 = tensor.extract %t[offsets...]
//   : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
// ```
//
// becomes:
//
// ```mlir
// %1 = tensor.extract_slice %arg0
//   [offsets...] [1..., lweDimension+1] [1...]
//   : tensor<...xlweDimension+1,i64> to
//     tensor<1...xlweDimension+1,i64>
// %0 = linalg.tensor_collapse_shape %0 [[...]]  :
// tensor<1x1xlweDimension+1xi64> into tensor<lweDimension+1xi64>
// ```
//
// TODO: since they are a bug on lowering extract_slice with rank reduction we
// add a linalg.tensor_collapse_shape after the extract_slice without rank
// reduction. See
// https://github.com/zama-ai/concrete-compiler-internal/issues/396.
struct ExtractOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::ExtractOp> {
  ExtractOpPattern(::mlir::MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::ExtractOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::ExtractOp extractOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto lweResultTy =
        extractOp.result()
            .getType()
            .dyn_cast_or_null<
                mlir::concretelang::Concrete::LweCiphertextType>();
    if (lweResultTy == nullptr) {
      return mlir::failure();
    }

    auto newResultTy =
        converter.convertType(lweResultTy).cast<mlir::RankedTensorType>();
    auto rankOfResult = extractOp.indices().size() + 1;

    // [min..., 0] for static_offsets ()
    mlir::SmallVector<mlir::Attribute> staticOffsets(
        rankOfResult,
        rewriter.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
    staticOffsets[staticOffsets.size() - 1] = rewriter.getI64IntegerAttr(0);

    // [1..., lweDimension+1] for static_sizes
    mlir::SmallVector<mlir::Attribute> staticSizes(
        rankOfResult, rewriter.getI64IntegerAttr(1));
    staticSizes[staticSizes.size() - 1] = rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1));

    // [1...] for static_strides
    mlir::SmallVector<mlir::Attribute> staticStrides(
        rankOfResult, rewriter.getI64IntegerAttr(1));

    // replace tensor.extract_slice to the new one
    mlir::SmallVector<int64_t> extractedSliceShape(
        extractOp.indices().size() + 1, 0);
    extractedSliceShape.reserve(extractOp.indices().size() + 1);
    for (size_t i = 0; i < extractedSliceShape.size() - 1; i++) {
      extractedSliceShape[i] = 1;
    }
    extractedSliceShape[extractedSliceShape.size() - 1] =
        newResultTy.getDimSize(0);

    auto extractedSliceType =
        mlir::RankedTensorType::get(extractedSliceShape, rewriter.getI64Type());
    auto extractedSlice = rewriter.create<mlir::tensor::ExtractSliceOp>(
        extractOp.getLoc(), extractedSliceType, extractOp.tensor(),
        extractOp.indices(), mlir::SmallVector<mlir::Value>{},
        mlir::SmallVector<mlir::Value>{}, rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));
    mlir::ReassociationIndices reassociation;
    for (int64_t i = 0; i < extractedSliceType.getRank(); i++) {
      reassociation.push_back(i);
    }
    rewriter.replaceOpWithNewOp<mlir::linalg::TensorCollapseShapeOp>(
        extractOp, newResultTy, extractedSlice,
        mlir::SmallVector<mlir::ReassociationIndices>{reassociation});
    return ::mlir::success();
  };
};

// This rewrite pattern transforms any instance of
// `tensor.insert_slice` operators that operates on tensor of lwe ciphertext.
//
// Example:
//
// ```mlir
// %0 = tensor.insert_slice %arg1
//        into %arg0[offsets...] [sizes...] [strides...]
//        : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> into
//          tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
// ```
//
// becomes:
//
// ```mlir
// %0 = tensor.insert_slice %arg1
//        into %arg0[offsets..., 0] [sizes..., lweDimension+1] [strides..., 1]
//        : tensor<...xlweDimension+1xi64> into
//          tensor<...xlweDimension+1xi64>
// ```
struct InsertSliceOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::InsertSliceOp> {
  InsertSliceOpPattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::InsertSliceOp>(context,
                                                              benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertSliceOp insertSliceOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = insertSliceOp.result().getType();

    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    // add 0 to static_offsets
    mlir::SmallVector<mlir::Attribute> staticOffsets;
    staticOffsets.append(insertSliceOp.static_offsets().begin(),
                         insertSliceOp.static_offsets().end());
    staticOffsets.push_back(rewriter.getI64IntegerAttr(0));

    // add lweDimension+1 to static_sizes
    mlir::SmallVector<mlir::Attribute> staticSizes;
    staticSizes.append(insertSliceOp.static_sizes().begin(),
                       insertSliceOp.static_sizes().end());
    staticSizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // add 1 to the strides
    mlir::SmallVector<mlir::Attribute> staticStrides;
    staticStrides.append(insertSliceOp.static_strides().begin(),
                         insertSliceOp.static_strides().end());
    staticStrides.push_back(rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        insertSliceOp, newResultTy, insertSliceOp.source(),
        insertSliceOp.dest(), insertSliceOp.offsets(), insertSliceOp.sizes(),
        insertSliceOp.strides(), rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

    return ::mlir::success();
  };
};

// This rewrite pattern transforms any instance of
// `tensor.from_elements` operators that operates on tensor of lwe ciphertext.
//
// Example:
//
// ```mlir
// %0 = tensor.from_elements %e0, ..., %e(n-1)
//        : tensor<Nx!Concrete.lwe_ciphertext<lweDim,p>>
// ```
//
// becomes:
//
// ```mlir
// %m = memref.alloc() : memref<NxlweDim+1xi64>
// %s0 = memref.subview %m[0, 0][1, lweDim+1][1, 1] : memref<lweDim+1xi64>
// %m0 = memref.buffer_cast %e0 : memref<lweDim+1xi64>
// memref.copy %m0, s0 : memref<lweDim+1xi64> to memref<lweDim+1xi64>
// ...
// %s(n-1) = memref.subview %m[(n-1), 0][1, lweDim+1][1, 1]
//             : memref<lweDim+1xi64>
// %m(n-1) = memref.buffer_cast %e(n-1) : memref<lweDim+1xi64>
// memref.copy %e(n-1), s(n-1)
//   : memref<lweDim+1xi64> to memref<lweDim+1xi64>
// %0 = memref.tensor_load %m : memref<NxlweDim+1xi64>
// ```
struct FromElementsOpPattern
    : public mlir::OpRewritePattern<mlir::tensor::FromElementsOp> {
  FromElementsOpPattern(::mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::FromElementsOp>(context,
                                                               benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::FromElementsOp fromElementsOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    auto resultTy = fromElementsOp.result().getType();
    if (converter.isLegal(resultTy)) {
      return mlir::failure();
    }
    auto eltResultTy =
        resultTy.cast<mlir::RankedTensorType>()
            .getElementType()
            .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto newTensorResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();
    auto newMemrefResultTy = mlir::MemRefType::get(
        newTensorResultTy.getShape(), newTensorResultTy.getElementType());

    // %m = memref.alloc() : memref<NxlweDim+1xi64>
    auto mOp = rewriter.create<mlir::memref::AllocOp>(fromElementsOp.getLoc(),
                                                      newMemrefResultTy);

    // for i = 0 to n-1
    // %si = memref.subview %m[i, 0][1, lweDim+1][1, 1] : memref<lweDim+1xi64>
    // %mi = memref.buffer_cast %ei : memref<lweDim+1xi64>
    // memref.copy %mi, si : memref<lweDim+1xi64> to memref<lweDim+1xi64>
    auto subviewResultTy = mlir::MemRefType::get(
        {eltResultTy.getDimension() + 1}, newMemrefResultTy.getElementType());
    auto offset = 0;
    for (auto eiOp : fromElementsOp.elements()) {
      mlir::SmallVector<mlir::Attribute, 2> staticOffsets{
          rewriter.getI64IntegerAttr(offset), rewriter.getI64IntegerAttr(0)};
      mlir::SmallVector<mlir::Attribute, 2> staticSizes{
          rewriter.getI64IntegerAttr(1),
          rewriter.getI64IntegerAttr(eltResultTy.getDimension() + 1)};
      mlir::SmallVector<mlir::Attribute, 2> staticStrides{
          rewriter.getI64IntegerAttr(1), rewriter.getI64IntegerAttr(1)};
      auto siOp = rewriter.create<mlir::memref::SubViewOp>(
          fromElementsOp.getLoc(), subviewResultTy, mOp, mlir::ValueRange{},
          mlir::ValueRange{}, mlir::ValueRange{},
          rewriter.getArrayAttr(staticOffsets),
          rewriter.getArrayAttr(staticSizes),
          rewriter.getArrayAttr(staticStrides));
      auto miOp = rewriter.create<mlir::memref::BufferCastOp>(
          fromElementsOp.getLoc(), subviewResultTy, eiOp);
      rewriter.create<mlir::memref::CopyOp>(fromElementsOp.getLoc(), miOp,
                                            siOp);
      offset++;
    }

    // Go back to tensor world
    // %0 = memref.tensor_load %m : memref<NxlweDim+1xi64>
    rewriter.replaceOpWithNewOp<mlir::memref::TensorLoadOp>(fromElementsOp,
                                                            mOp);

    return ::mlir::success();
  };
};

// This template rewrite pattern transforms any instance of
// `ShapeOp` operators that operates on tensor of lwe ciphertext by adding the
// lwe size as a size of the tensor result and by adding a trivial reassociation
// at the end of the reassociations map.
//
// Example:
//
// ```mlir
// %0 = "ShapeOp" %arg0 [reassocations...]
//        : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> into
//          tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
// ```
//
// becomes:
//
// ```mlir
// %0 = "ShapeOp" %arg0 [reassociations..., [inRank or outRank]]
//        : tensor<...xlweDimesion+1xi64> into
//          tensor<...xlweDimesion+1xi64>
// ```
template <typename ShapeOp, bool inRank>
struct TensorShapeOpPattern : public mlir::OpRewritePattern<ShapeOp> {
  TensorShapeOpPattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<ShapeOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(ShapeOp shapeOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy = shapeOp.result().getType();

    auto newResultTy =
        ((mlir::Type)converter.convertType(resultTy)).cast<mlir::MemRefType>();

    // add [rank] to reassociations
    auto oldReassocs = shapeOp.getReassociationIndices();
    mlir::SmallVector<mlir::ReassociationIndices> newReassocs;
    newReassocs.append(oldReassocs.begin(), oldReassocs.end());
    mlir::ReassociationIndices lweAssoc;
    auto reassocTy =
        ((mlir::Type)converter.convertType(
             (inRank ? shapeOp.src() : shapeOp.result()).getType()))
            .cast<mlir::MemRefType>();
    lweAssoc.push_back(reassocTy.getRank());
    newReassocs.push_back(lweAssoc);

    rewriter.replaceOpWithNewOp<ShapeOp>(shapeOp, newResultTy, shapeOp.src(),
                                         newReassocs);
    return ::mlir::success();
  };
};

// Add the instantiated TensorShapeOpPattern rewrite pattern with the `ShapeOp`
// to the patterns set and populate the conversion target.
template <typename ShapeOp, bool inRank>
void insertTensorShapeOpPattern(mlir::MLIRContext &context,
                                mlir::RewritePatternSet &patterns,
                                mlir::ConversionTarget &target) {
  patterns.insert<TensorShapeOpPattern<ShapeOp, inRank>>(&context);
  target.addDynamicallyLegalOp<ShapeOp>([&](ShapeOp op) {
    ConcreteToBConcreteTypeConverter converter;
    return converter.isLegal(op.result().getType());
  });
}

// This template rewrite pattern transforms any instance of
// `MemrefOp` operators that returns a memref of lwe ciphertext to the same
// operator but which returns the bufferized lwe ciphertext.
//
// Example:
//
// ```mlir
// %0 = "MemrefOp"(...) : ... -> memref<...x!Concrete.lwe_ciphertext<lweDim,p>>
// ```
//
// becomes:
//
// ```mlir
// %0 = "MemrefOp"(...) : ... -> memref<...xlweDim+1xi64>
// ```
template <typename MemrefOp>
struct MemrefOpPattern : public mlir::OpRewritePattern<MemrefOp> {
  MemrefOpPattern(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<MemrefOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(MemrefOp memrefOp,
                  mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    mlir::SmallVector<mlir::Type, 1> convertedTypes;
    if (converter.convertTypes(memrefOp->getResultTypes(), convertedTypes)
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<MemrefOp>(memrefOp, convertedTypes,
                                          memrefOp->getOperands(),
                                          memrefOp->getAttrs());
    return ::mlir::success();
  };
};

// Add the instantiated MemrefOpPattern rewrite pattern with the `MemrefOp`
// to the patterns set and populate the conversion target.
template <typename... MemrefOp>
void insertMemrefOpPattern(mlir::MLIRContext &context,
                           mlir::RewritePatternSet &patterns,
                           mlir::ConversionTarget &target) {
  (void)std::initializer_list<int>{
      0, (patterns.insert<MemrefOpPattern<MemrefOp>>(&context),
          target.addDynamicallyLegalOp<MemrefOp>([&](MemrefOp op) {
            ConcreteToBConcreteTypeConverter converter;
            return converter.isLegal(op->getResultTypes());
          }),
          0)...};
}

// cc from Loops.cpp
static mlir::SmallVector<mlir::Value>
makeCanonicalAffineApplies(mlir::OpBuilder &b, mlir::Location loc,
                           mlir::AffineMap map,
                           mlir::ArrayRef<mlir::Value> vals) {
  if (map.isEmpty())
    return {};

  assert(map.getNumInputs() == vals.size());
  mlir::SmallVector<mlir::Value> res;
  res.reserve(map.getNumResults());
  auto dims = map.getNumDims();
  for (auto e : map.getResults()) {
    auto exprMap = mlir::AffineMap::get(dims, map.getNumSymbols(), e);
    mlir::SmallVector<mlir::Value> operands(vals.begin(), vals.end());
    canonicalizeMapAndOperands(&exprMap, &operands);
    res.push_back(b.create<mlir::AffineApplyOp>(loc, exprMap, operands));
  }
  return res;
}

static std::pair<mlir::Value, mlir::Value>
makeOperandLoadOrSubview(mlir::OpBuilder &builder, mlir::Location loc,
                         mlir::ArrayRef<mlir::Value> allIvs,
                         mlir::linalg::LinalgOp linalgOp,
                         mlir::OpOperand *operand) {
  ConcreteToBConcreteTypeConverter converter;

  mlir::Value opVal = operand->get();
  mlir::MemRefType opTy = opVal.getType().cast<mlir::MemRefType>();

  if (auto lweType =
          opTy.getElementType()
              .dyn_cast_or_null<
                  mlir::concretelang::Concrete::LweCiphertextType>()) {
    // For memref of ciphertexts operands create the inner memref
    // subview to the ciphertext, and go back to the tensor type as BConcrete
    // operators works with tensor.
    // %op : memref<dim...xConcrete.lwe_ciphertext<lweDim,p>>
    // %opInner = memref.subview %opInner[offsets...][1...][1,...]
    //   : memref<...xConcrete.lwe_ciphertext<lweDim,p>> to
    //     memref<Concrete.lwe_ciphertext<lweDim,p>>

    auto tensorizedLweTy =
        converter.convertType(lweType).cast<mlir::RankedTensorType>();
    auto subviewResultTy = mlir::MemRefType::get(
        tensorizedLweTy.getShape(), tensorizedLweTy.getElementType());
    auto offsets = makeCanonicalAffineApplies(
        builder, loc, linalgOp.getTiedIndexingMap(operand), allIvs);
    mlir::SmallVector<mlir::Attribute> staticOffsets(
        opTy.getRank(),
        builder.getI64IntegerAttr(std::numeric_limits<int64_t>::min()));
    mlir::SmallVector<mlir::Attribute> staticSizes(
        opTy.getRank(), builder.getI64IntegerAttr(1));
    mlir::SmallVector<mlir::Attribute> staticStrides(
        opTy.getRank(), builder.getI64IntegerAttr(1));

    auto subViewOp = builder.create<mlir::memref::SubViewOp>(
        loc, subviewResultTy, opVal, offsets, mlir::ValueRange{},
        mlir::ValueRange{}, builder.getArrayAttr(staticOffsets),
        builder.getArrayAttr(staticSizes), builder.getArrayAttr(staticStrides));
    return std::pair<mlir::Value, mlir::Value>(
        subViewOp, builder.create<mlir::memref::TensorLoadOp>(loc, subViewOp));
  } else {
    // For memref of non ciphertexts load the value from the memref.
    // with %op : memref<dim...xip>
    // %opInner = memref.load %op[offsets...] : memref<dim...xip>
    auto offsets = makeCanonicalAffineApplies(
        builder, loc, linalgOp.getTiedIndexingMap(operand), allIvs);
    return std::pair<mlir::Value, mlir::Value>(
        nullptr,
        builder.create<mlir::memref::LoadOp>(loc, operand->get(), offsets));
  }
}

static void
inlineRegionAndEmitTensorStore(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::linalg::LinalgOp linalgOp,
                               llvm::ArrayRef<mlir::Value> indexedValues,
                               mlir::ValueRange outputBuffers) {
  // Clone the block with the new operands
  auto &block = linalgOp->getRegion(0).front();
  mlir::BlockAndValueMapping map;
  map.map(block.getArguments(), indexedValues);
  for (auto &op : block.without_terminator()) {
    auto *newOp = builder.clone(op, map);
    map.map(op.getResults(), newOp->getResults());
  }
  // Create memref.tensor_store operation for each terminator operands
  auto *terminator = block.getTerminator();
  for (mlir::OpOperand &operand : terminator->getOpOperands()) {
    mlir::Value toStore = map.lookupOrDefault(operand.get());
    builder.create<mlir::memref::TensorStoreOp>(
        loc, toStore, outputBuffers[operand.getOperandNumber()]);
  }
}

template <typename LoopType>
class LinalgRewritePattern
    : public mlir::OpInterfaceConversionPattern<mlir::linalg::LinalgOp> {
public:
  using mlir::OpInterfaceConversionPattern<
      mlir::linalg::LinalgOp>::OpInterfaceConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::LinalgOp linalgOp,
                  mlir::ArrayRef<mlir::Value> operands,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    assert(linalgOp.hasBufferSemantics() &&
           "expected linalg op with buffer semantics");

    auto loopRanges = linalgOp.createLoopRanges(rewriter, linalgOp.getLoc());
    auto iteratorTypes =
        llvm::to_vector<4>(linalgOp.iterator_types().getValue());

    mlir::SmallVector<mlir::Value> allIvs;
    mlir::linalg::GenerateLoopNest<LoopType>::doit(
        rewriter, linalgOp.getLoc(), loopRanges, linalgOp, iteratorTypes,
        [&](mlir::OpBuilder &builder, mlir::Location loc, mlir::ValueRange ivs,
            mlir::ValueRange operandValuesToUse) -> mlir::scf::ValueVector {
          // Keep indexed values to replace the linalg.generic block arguments
          // by them
          mlir::SmallVector<mlir::Value> indexedValues;
          indexedValues.reserve(linalgOp.getNumInputsAndOutputs());
          assert(
              operandValuesToUse == linalgOp->getOperands() &&
              "expect operands are captured and not passed by loop argument");
          allIvs.append(ivs.begin(), ivs.end());

          // For all input operands create the inner operand
          for (mlir::OpOperand *inputOperand : linalgOp.getInputOperands()) {
            auto innerOperand = makeOperandLoadOrSubview(
                builder, loc, allIvs, linalgOp, inputOperand);
            indexedValues.push_back(innerOperand.second);
          }

          // For all output operands create the inner operand
          assert(linalgOp.getOutputOperands() ==
                     linalgOp.getOutputBufferOperands() &&
                 "expect only memref as output operands");
          mlir::SmallVector<mlir::Value> outputBuffers;
          for (mlir::OpOperand *outputOperand : linalgOp.getOutputOperands()) {
            auto innerOperand = makeOperandLoadOrSubview(
                builder, loc, allIvs, linalgOp, outputOperand);
            indexedValues.push_back(innerOperand.second);
            assert(innerOperand.first != nullptr &&
                   "Expected a memref subview as output buffer");
            outputBuffers.push_back(innerOperand.first);
          }
          // Finally inline the linalgOp region
          inlineRegionAndEmitTensorStore(builder, loc, linalgOp, indexedValues,
                                         outputBuffers);

          return mlir::scf::ValueVector{};
        });
    rewriter.eraseOp(linalgOp);
    return mlir::success();
  };
};

void ConcreteToBConcretePass::runOnOperation() {
  auto op = this->getOperation();

  // First of all we transform LinalgOp that work on tensor of ciphertext to
  // work on memref.
  {
    mlir::ConversionTarget target(getContext());
    mlir::BufferizeTypeConverter converter;

    // Mark all Standard operations legal.
    target
        .addLegalDialect<mlir::arith::ArithmeticDialect, mlir::AffineDialect,
                         mlir::memref::MemRefDialect, mlir::StandardOpsDialect,
                         mlir::tensor::TensorDialect>();

    // Mark all Linalg operations illegal as long as they work on encrypted
    // tensors.
    target.addDynamicallyLegalOp<mlir::linalg::GenericOp, mlir::linalg::YieldOp,
                                 mlir::linalg::CopyOp>(
        [&](mlir::Operation *op) { return converter.isLegal(op); });

    mlir::RewritePatternSet patterns(&getContext());
    mlir::linalg::populateLinalgBufferizePatterns(converter, patterns);
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  // Then convert ciphertext to tensor or add a dimension to tensor of
  // ciphertext and memref of ciphertext
  {
    mlir::ConversionTarget target(getContext());
    ConcreteToBConcreteTypeConverter converter;
    mlir::OwningRewritePatternList patterns(&getContext());

    // All BConcrete ops are legal after the conversion
    target.addLegalDialect<mlir::concretelang::BConcrete::BConcreteDialect>();

    // Add Concrete ops are illegal after the conversion unless those which are
    // explicitly marked as legal (more or less operators that didn't work on
    // ciphertexts)
    target.addIllegalDialect<mlir::concretelang::Concrete::ConcreteDialect>();
    target.addLegalOp<mlir::concretelang::Concrete::EncodeIntOp>();
    target.addLegalOp<mlir::concretelang::Concrete::GlweFromTable>();
    target.addLegalOp<mlir::concretelang::Concrete::IntToCleartextOp>();

    // Add patterns to convert the zero ops to tensor.generate
    patterns
        .insert<ZeroOpPattern<mlir::concretelang::Concrete::ZeroTensorLWEOp>,
                ZeroOpPattern<mlir::concretelang::Concrete::ZeroLWEOp>>(
            &getContext());
    target.addLegalOp<mlir::tensor::GenerateOp, mlir::tensor::YieldOp>();

    // Add patterns to trivialy convert Concrete op to the equivalent
    // BConcrete op
    target.addLegalOp<mlir::linalg::InitTensorOp>();
    patterns.insert<
        LowToBConcrete<mlir::concretelang::Concrete::AddLweCiphertextsOp,
                       mlir::concretelang::BConcrete::AddLweBuffersOp>,
        LowToBConcrete<
            mlir::concretelang::Concrete::AddPlaintextLweCiphertextOp,
            mlir::concretelang::BConcrete::AddPlaintextLweBufferOp>,
        LowToBConcrete<
            mlir::concretelang::Concrete::MulCleartextLweCiphertextOp,
            mlir::concretelang::BConcrete::MulCleartextLweBufferOp>,
        LowToBConcrete<
            mlir::concretelang::Concrete::MulCleartextLweCiphertextOp,
            mlir::concretelang::BConcrete::MulCleartextLweBufferOp>,
        LowToBConcrete<mlir::concretelang::Concrete::NegateLweCiphertextOp,
                       mlir::concretelang::BConcrete::NegateLweBufferOp>,
        LowToBConcrete<mlir::concretelang::Concrete::KeySwitchLweOp,
                       mlir::concretelang::BConcrete::KeySwitchLweBufferOp>,
        LowToBConcrete<mlir::concretelang::Concrete::BootstrapLweOp,
                       mlir::concretelang::BConcrete::BootstrapLweBufferOp>>(
        &getContext());

    // Add patterns to rewrite tensor operators that works on encrypted tensors
    patterns.insert<ExtractSliceOpPattern, ExtractOpPattern,
                    InsertSliceOpPattern, FromElementsOpPattern>(&getContext());
    target.addDynamicallyLegalOp<
        mlir::tensor::ExtractSliceOp, mlir::tensor::ExtractOp,
        mlir::tensor::InsertSliceOp, mlir::tensor::FromElementsOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResult(0).getType());
        });
    target.addLegalOp<mlir::memref::CopyOp,
                      mlir::linalg::TensorCollapseShapeOp>();

    // Add patterns to rewrite some of memref ops that was introduced by the
    // linalg bufferization of encrypted tensor (first conversion of this pass)
    insertTensorShapeOpPattern<mlir::memref::ExpandShapeOp, true>(
        getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::memref::CollapseShapeOp, false>(
        getContext(), patterns, target);

    // Add patterns to rewrite linalg op to nested loops with views on
    // ciphertexts
    patterns.insert<LinalgRewritePattern<mlir::scf::ForOp>>(converter,
                                                            &getContext());
    target.addLegalOp<mlir::arith::ConstantOp, mlir::scf::ForOp,
                      mlir::scf::YieldOp, mlir::AffineApplyOp,
                      mlir::memref::SubViewOp, mlir::memref::LoadOp,
                      mlir::memref::TensorStoreOp>();

    // Add patterns to do the conversion of func
    mlir::populateFuncOpTypeConversionPattern(patterns, converter);
    target.addDynamicallyLegalOp<mlir::FuncOp>([&](mlir::FuncOp funcOp) {
      return converter.isSignatureLegal(funcOp.getType()) &&
             converter.isLegal(&funcOp.getBody());
    });

    // Add patterns to convert some memref operators that is generated by
    // previous step
    insertMemrefOpPattern<mlir::memref::AllocOp, mlir::memref::BufferCastOp,
                          mlir::memref::TensorLoadOp>(getContext(), patterns,
                                                      target);

    // Conversion of RT Dialect Ops
    patterns.add<mlir::concretelang::GenericTypeConverterPattern<
        mlir::concretelang::RT::DataflowTaskOp>>(patterns.getContext(),
                                                 converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowTaskOp>(target, converter);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>>
createConvertConcreteToBConcretePass() {
  return std::make_unique<ConcreteToBConcretePass>();
}
} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <algorithm>
#include <iostream>
#include <iterator>
#include <mlir/Dialect/Affine/IR/AffineOps.h>
#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Dialect/Tensor/IR/Tensor.h>
#include <mlir/IR/AffineExpr.h>
#include <mlir/IR/AffineMap.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/Support/LLVM.h>

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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/Function.h"

namespace {
struct ConcreteToBConcretePass
    : public ConcreteToBConcreteBase<ConcreteToBConcretePass> {
  void runOnOperation() final;
  ConcreteToBConcretePass() = delete;
  ConcreteToBConcretePass(bool loopParallelize)
      : loopParallelize(loopParallelize){};

private:
  bool loopParallelize;
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
    addConversion([&](mlir::concretelang::Concrete::PlaintextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::concretelang::Concrete::CleartextType type) {
      return mlir::IntegerType::get(type.getContext(), 64);
    });
    addConversion([&](mlir::concretelang::Concrete::LweCiphertextType type) {
      assert(type.getDimension() != -1);
      return mlir::RankedTensorType::get(
          {type.getDimension() + 1},
          mlir::IntegerType::get(type.getContext(), 64));
    });
    addConversion([&](mlir::concretelang::Concrete::GlweCiphertextType type) {
      assert(type.getGlweDimension() != -1);
      assert(type.getPolynomialSize() != -1);

      return mlir::RankedTensorType::get(
          {type.getPolynomialSize() * (type.getGlweDimension() + 1)},
          mlir::IntegerType::get(type.getContext(), 64));
    });
    addConversion([&](mlir::RankedTensorType type) {
      auto lwe = type.getElementType()
                     .dyn_cast_or_null<
                         mlir::concretelang::Concrete::LweCiphertextType>();
      if (lwe == nullptr) {
        return (mlir::Type)(type);
      }
      assert(lwe.getDimension() != -1);
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
      assert(lwe.getDimension() != -1);
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

struct ConcreteEncodeIntOpPattern
    : public mlir::OpRewritePattern<mlir::concretelang::Concrete::EncodeIntOp> {
  ConcreteEncodeIntOpPattern(mlir::MLIRContext *context,
                             mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::EncodeIntOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::EncodeIntOp op,
                  mlir::PatternRewriter &rewriter) const override {
    {
      mlir::Value castedInt = rewriter.create<mlir::arith::ExtUIOp>(
          op.getLoc(), rewriter.getIntegerType(64), op->getOperands().front());
      mlir::Value constantShiftOp = rewriter.create<mlir::arith::ConstantOp>(
          op.getLoc(), rewriter.getI64IntegerAttr(64 - op.getType().getP()));

      mlir::Type resultType = rewriter.getIntegerType(64);
      rewriter.replaceOpWithNewOp<mlir::arith::ShLIOp>(
          op, resultType, castedInt, constantShiftOp);
    }
    return mlir::success();
  };
};

struct ConcreteIntToCleartextOpPattern
    : public mlir::OpRewritePattern<
          mlir::concretelang::Concrete::IntToCleartextOp> {
  ConcreteIntToCleartextOpPattern(mlir::MLIRContext *context,
                                  mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::concretelang::Concrete::IntToCleartextOp>(
            context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::IntToCleartextOp op,
                  mlir::PatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<mlir::arith::ExtUIOp>(
        op, rewriter.getIntegerType(64), op->getOperands().front());
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of `Concrete.zero_tensor`
/// operators.
///
/// Example:
///
/// ```mlir
/// %0 = "Concrete.zero_tensor" () :
/// tensor<...x!Concrete.lwe_ciphertext<lweDim,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %0 = tensor.generate {
///   ^bb0(... : index):
///     %c0 = arith.constant 0 : i64
///     tensor.yield %z
/// }: tensor<...xlweDim+1xi64>
/// i64>
/// ```
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

/// This template rewrite pattern transforms any instance of
/// `ConcreteOp` to an instance of `BConcreteOp`.
///
/// Example:
///
///   %0 = "ConcreteOp"(%arg0, ...) :
///     (!Concrete.lwe_ciphertext<lwe_dimension, p>, ...) ->
///     (!Concrete.lwe_ciphertext<lwe_dimension, p>)
///
/// becomes:
///
///   %0 = "BConcreteOp"(%arg0, ...) : (tensor<dimension+1, i64>>, ..., ) ->
///   (tensor<dimension+1, i64>>)
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

    llvm::ArrayRef<::mlir::NamedAttribute> attributes =
        concreteOp.getOperation()->getAttrs();

    BConcreteOp bConcreteOp = rewriter.replaceOpWithNewOp<BConcreteOp>(
        concreteOp, newResultTy, concreteOp.getOperation()->getOperands(),
        attributes);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, bConcreteOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of
/// `Concrete.glwe_from_table` operators.
///
/// Example:
///
/// ```mlir
/// %0 = "Concrete.glwe_from_table"(%tlu)
///   : (tensor<$Dxi64>) ->
///   !Concrete.glwe_ciphertext<$polynomialSize,$glweDimension,$p>
/// ```
///
/// with $D = 2^$p
///
/// becomes:
///
/// ```mlir
/// %0 = linalg.init_tensor [polynomialSize*(glweDimension+1)]
///        : tensor<polynomialSize*(glweDimension+1), i64>
/// "BConcrete.fill_glwe_from_table" : (%0, polynomialSize, glweDimension, %tlu)
///   : tensor<polynomialSize*(glweDimension+1), i64>, i64, i64, tensor<$Dxi64>
/// ```
struct GlweFromTablePattern : public mlir::OpRewritePattern<
                                  mlir::concretelang::Concrete::GlweFromTable> {
  GlweFromTablePattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::Concrete::GlweFromTable>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::GlweFromTable op,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    auto resultTy =
        op.result()
            .getType()
            .cast<mlir::concretelang::Concrete::GlweCiphertextType>();

    auto newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();
    // %0 = linalg.init_tensor [polynomialSize*(glweDimension+1)]
    //        : tensor<polynomialSize*(glweDimension+1), i64>
    mlir::Value init =
        rewriter.replaceOpWithNewOp<mlir::bufferization::AllocTensorOp>(
            op, newResultTy, mlir::ValueRange{});

    // "BConcrete.fill_glwe_from_table" : (%0, polynomialSize, glweDimension,
    // %tlu)
    auto polySize = resultTy.getPolynomialSize();
    auto glweDimension = resultTy.getGlweDimension();
    auto outPrecision = resultTy.getP();

    rewriter.create<mlir::concretelang::BConcrete::FillGlweFromTable>(
        op.getLoc(), init, glweDimension, polySize, outPrecision, op.table());

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of
/// `tensor.extract_slice` operators that operates on tensor of lwe ciphertext.
///
/// Example:
///
/// ```mlir
/// %0 = tensor.extract_slice %arg0
///   [offsets...] [sizes...] [strides...]
///   : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> to
///     tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %0 = tensor.extract_slice %arg0
///   [offsets..., 0] [sizes..., lweDimension+1] [strides..., 1]
///   : tensor<...xlweDimension+1,i64> to
///     tensor<...xlweDimension+1,i64>
/// ```
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
    mlir::tensor::ExtractSliceOp extractOp =
        rewriter.replaceOpWithNewOp<mlir::tensor::ExtractSliceOp>(
            extractSliceOp, newResultTy, extractSliceOp.source(),
            extractSliceOp.offsets(), extractSliceOp.sizes(),
            extractSliceOp.strides(), rewriter.getArrayAttr(staticOffsets),
            rewriter.getArrayAttr(staticSizes),
            rewriter.getArrayAttr(staticStrides));

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, extractOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of
/// `tensor.extract` operators that operates on tensor of lwe ciphertext.
///
/// Example:
///
/// ```mlir
/// %0 = tensor.extract %t[offsets...]
///   : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %1 = tensor.extract_slice %arg0
///   [offsets...] [1..., lweDimension+1] [1...]
///   : tensor<...xlweDimension+1,i64> to
///     tensor<1...xlweDimension+1,i64>
/// %0 = linalg.tensor_collapse_shape %0 [[...]]  :
/// tensor<1x1xlweDimension+1xi64> into tensor<lweDimension+1xi64>
/// ```
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

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, extractedSlice, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    mlir::ReassociationIndices reassociation;
    for (int64_t i = 0; i < extractedSliceType.getRank(); i++) {
      reassociation.push_back(i);
    }

    mlir::tensor::CollapseShapeOp collapseOp =
        rewriter.replaceOpWithNewOp<mlir::tensor::CollapseShapeOp>(
            extractOp, newResultTy, extractedSlice,
            mlir::SmallVector<mlir::ReassociationIndices>{reassociation});

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, collapseOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of
/// `tensor.insert_slice` operators that operates on tensor of lwe ciphertext.
///
/// Example:
///
/// ```mlir
/// %0 = tensor.insert_slice %arg1
///        into %arg0[offsets...] [sizes...] [strides...]
///        : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> into
///          tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %0 = tensor.insert_slice %arg1
///        into %arg0[offsets..., 0] [sizes..., lweDimension+1] [strides..., 1]
///        : tensor<...xlweDimension+1xi64> into
///          tensor<...xlweDimension+1xi64>
/// ```
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
    auto newOp = rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
        insertSliceOp, newResultTy, insertSliceOp.source(),
        insertSliceOp.dest(), insertSliceOp.offsets(), insertSliceOp.sizes(),
        insertSliceOp.strides(), rewriter.getArrayAttr(staticOffsets),
        rewriter.getArrayAttr(staticSizes),
        rewriter.getArrayAttr(staticStrides));

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, newOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of `tensor.insert`
/// operators that operates on an lwe ciphertexts to a
/// `tensor.insert_slice` op operating on the bufferized representation
/// of the ciphertext.
///
/// Example:
///
/// ```mlir
/// %0 = tensor.insert %arg1
///        into %arg0[offsets...]
///        : !Concrete.lwe_ciphertext<lweDimension,p> into
///          tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %0 = tensor.insert_slice %arg1
///        into %arg0[offsets..., 0] [sizes..., lweDimension+1] [strides..., 1]
///        : tensor<lweDimension+1xi64> into
///          tensor<...xlweDimension+1xi64>
/// ```
struct InsertOpPattern : public mlir::OpRewritePattern<mlir::tensor::InsertOp> {
  InsertOpPattern(::mlir::MLIRContext *context,
                  mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::tensor::InsertOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::tensor::InsertOp insertOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    mlir::Type resultTy = insertOp.result().getType();
    mlir::RankedTensorType newResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    // add 0 to static_offsets
    mlir::SmallVector<mlir::OpFoldResult> offsets;
    offsets.append(insertOp.indices().begin(), insertOp.indices().end());
    offsets.push_back(rewriter.getIndexAttr(0));

    // Inserting a smaller tensor into a (potentially) bigger one. Set
    // dimensions for all leading dimensions of the target tensor not
    // present in the source to 1.
    mlir::SmallVector<mlir::OpFoldResult> sizes(insertOp.indices().size(),
                                                rewriter.getI64IntegerAttr(1));

    // Add size for the bufferized source element
    sizes.push_back(rewriter.getI64IntegerAttr(
        newResultTy.getDimSize(newResultTy.getRank() - 1)));

    // Set stride of all dimensions to 1
    mlir::SmallVector<mlir::OpFoldResult> strides(
        newResultTy.getRank(), rewriter.getI64IntegerAttr(1));

    // replace tensor.insert_slice with the new one
    mlir::tensor::InsertSliceOp insertSliceOp =
        rewriter.replaceOpWithNewOp<mlir::tensor::InsertSliceOp>(
            insertOp, insertOp.getOperand(0), insertOp.dest(), offsets, sizes,
            strides);

    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, insertSliceOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of
/// `tensor.from_elements` operators that operates on tensor of lwe ciphertext.
///
/// Example:
///
/// ```mlir
/// %0 = tensor.from_elements %e0, ..., %e(n-1)
///        : tensor<Nx!Concrete.lwe_ciphertext<lweDim,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %m = memref.alloc() : memref<NxlweDim+1xi64>
/// %s0 = memref.subview %m[0, 0][1, lweDim+1][1, 1] : memref<lweDim+1xi64>
/// %m0 = memref.buffer_cast %e0 : memref<lweDim+1xi64>
/// memref.copy %m0, s0 : memref<lweDim+1xi64> to memref<lweDim+1xi64>
/// ...
/// %s(n-1) = memref.subview %m[(n-1), 0][1, lweDim+1][1, 1]
///             : memref<lweDim+1xi64>
/// %m(n-1) = memref.buffer_cast %e(n-1) : memref<lweDim+1xi64>
/// memref.copy %e(n-1), s(n-1)
///   : memref<lweDim+1xi64> to memref<lweDim+1xi64>
/// %0 = memref.tensor_load %m : memref<NxlweDim+1xi64>
/// ```
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

    auto newTensorResultTy =
        converter.convertType(resultTy).cast<mlir::RankedTensorType>();

    mlir::Value tensor = rewriter.create<mlir::bufferization::AllocTensorOp>(
        fromElementsOp.getLoc(), newTensorResultTy, mlir::ValueRange{});

    llvm::SmallVector<mlir::OpFoldResult> sizes(1,
                                                rewriter.getI64IntegerAttr(1));
    std::transform(newTensorResultTy.getShape().begin() + 1,
                   newTensorResultTy.getShape().end(),
                   std::back_inserter(sizes),
                   [&](auto v) { return rewriter.getI64IntegerAttr(v); });

    llvm::SmallVector<mlir::OpFoldResult> oneStrides(
        newTensorResultTy.getShape().size(), rewriter.getI64IntegerAttr(1));

    llvm::SmallVector<mlir::OpFoldResult> offsets(
        newTensorResultTy.getRank(), rewriter.getI64IntegerAttr(0));

    for (auto elt : llvm::enumerate(fromElementsOp.elements())) {
      offsets[0] = rewriter.getI64IntegerAttr(elt.index());

      mlir::tensor::InsertSliceOp insOp =
          rewriter.create<mlir::tensor::InsertSliceOp>(
              fromElementsOp.getLoc(),
              /* src: */ elt.value(),
              /* dst: */ tensor,
              /* offs: */ offsets,
              /* sizes: */ sizes,
              /* strides: */ oneStrides);

      mlir::concretelang::convertOperandAndResultTypes(
          rewriter, insOp, [&](mlir::MLIRContext *, mlir::Type t) {
            return converter.convertType(t);
          });

      tensor = insOp.getResult();
    }

    rewriter.replaceOp(fromElementsOp, tensor);
    return ::mlir::success();
  };
};

/// This template rewrite pattern transforms any instance of
/// `ShapeOp` operators that operates on tensor of lwe ciphertext by adding the
/// lwe size as a size of the tensor result and by adding a trivial
/// reassociation at the end of the reassociations map.
///
/// Example:
///
/// ```mlir
/// %0 = "ShapeOp" %arg0 [reassocations...]
///        : tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>> into
///          tensor<...x!Concrete.lwe_ciphertext<lweDimension,p>>
/// ```
///
/// becomes:
///
/// ```mlir
/// %0 = "ShapeOp" %arg0 [reassociations..., [inRank or outRank]]
///        : tensor<...xlweDimesion+1xi64> into
///          tensor<...xlweDimesion+1xi64>
/// ```
template <typename ShapeOp, typename VecTy, bool inRank>
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
        ((mlir::Type)converter.convertType(resultTy)).cast<VecTy>();

    // add [rank] to reassociations
    auto oldReassocs = shapeOp.getReassociationIndices();
    mlir::SmallVector<mlir::ReassociationIndices> newReassocs;
    newReassocs.append(oldReassocs.begin(), oldReassocs.end());
    mlir::ReassociationIndices lweAssoc;
    auto reassocTy =
        ((mlir::Type)converter.convertType(
             (inRank ? shapeOp.src() : shapeOp.result()).getType()))
            .cast<VecTy>();
    lweAssoc.push_back(reassocTy.getRank() - 1);
    newReassocs.push_back(lweAssoc);

    ShapeOp op = rewriter.replaceOpWithNewOp<ShapeOp>(
        shapeOp, newResultTy, shapeOp.src(), newReassocs);

    // fix operand types
    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, op, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });

    return ::mlir::success();
  };
};

/// Add the instantiated TensorShapeOpPattern rewrite pattern with the `ShapeOp`
/// to the patterns set and populate the conversion target.
template <typename ShapeOp, typename VecTy, bool inRank>
void insertTensorShapeOpPattern(mlir::MLIRContext &context,
                                mlir::RewritePatternSet &patterns,
                                mlir::ConversionTarget &target) {
  patterns.insert<TensorShapeOpPattern<ShapeOp, VecTy, inRank>>(&context);
  target.addDynamicallyLegalOp<ShapeOp>([&](mlir::Operation *op) {
    ConcreteToBConcreteTypeConverter converter;
    return converter.isLegal(op->getResultTypes()) &&
           converter.isLegal(op->getOperandTypes());
  });
}

/// Rewrites `bufferization.alloc_tensor` ops for which the converted type in
/// BConcrete is different from the original type.
///
/// Example:
///
/// ```
/// bufferization.alloc_tensor() : tensor<4x!Concrete.lwe_ciphertext<4096,6>>
/// ```
///
/// becomes:
///
/// ```
/// bufferization.alloc_tensor() : tensor<4x4097xi64>
/// ```
struct AllocTensorOpPattern
    : public mlir::OpRewritePattern<mlir::bufferization::AllocTensorOp> {
  AllocTensorOpPattern(::mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::bufferization::AllocTensorOp>(context,
                                                                     benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::bufferization::AllocTensorOp allocTensorOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;
    mlir::RankedTensorType resultTy =
        allocTensorOp.getType().dyn_cast<mlir::RankedTensorType>();

    if (!resultTy || !resultTy.hasStaticShape())
      return mlir::failure();

    mlir::RankedTensorType newResultTy =
        converter.convertType(resultTy).dyn_cast<mlir::RankedTensorType>();

    if (resultTy.getShape().size() != newResultTy.getShape().size()) {
      rewriter.replaceOpWithNewOp<mlir::bufferization::AllocTensorOp>(
          allocTensorOp, newResultTy, mlir::ValueRange{});
    }

    return ::mlir::success();
  };
};

struct ForOpPattern : public mlir::OpRewritePattern<mlir::scf::ForOp> {
  ForOpPattern(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::scf::ForOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::scf::ForOp forOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    ConcreteToBConcreteTypeConverter converter;

    // TODO: Check if there is a cleaner way to modify the types in
    // place through appropriate interfaces or by reconstructing the
    // ForOp with the right types.
    rewriter.updateRootInPlace(forOp, [&] {
      for (mlir::Value initArg : forOp.getInitArgs()) {
        mlir::Type convertedType = converter.convertType(initArg.getType());
        initArg.setType(convertedType);
      }

      for (mlir::Value &blockArg : forOp.getBody()->getArguments()) {
        mlir::Type convertedType = converter.convertType(blockArg.getType());
        blockArg.setType(convertedType);
      }

      for (mlir::OpResult result : forOp.getResults()) {
        mlir::Type convertedType = converter.convertType(result.getType());
        result.setType(convertedType);
      }
    });

    return ::mlir::success();
  };
};

void ConcreteToBConcretePass::runOnOperation() {
  auto op = this->getOperation();

  // Then convert ciphertext to tensor or add a dimension to tensor of
  // ciphertext and memref of ciphertext
  {
    mlir::ConversionTarget target(getContext());
    ConcreteToBConcreteTypeConverter converter;
    mlir::RewritePatternSet patterns(&getContext());

    // All BConcrete ops are legal after the conversion
    target.addLegalDialect<mlir::concretelang::BConcrete::BConcreteDialect>();

    // Add Concrete ops are illegal after the conversion
    target.addIllegalDialect<mlir::concretelang::Concrete::ConcreteDialect>();

    // Add patterns to convert cleartext and plaintext to i64
    patterns
        .insert<ConcreteEncodeIntOpPattern, ConcreteIntToCleartextOpPattern>(
            &getContext());
    target.addLegalDialect<mlir::arith::ArithmeticDialect>();

    // Add patterns to convert the zero ops to tensor.generate
    patterns
        .insert<ZeroOpPattern<mlir::concretelang::Concrete::ZeroTensorLWEOp>,
                ZeroOpPattern<mlir::concretelang::Concrete::ZeroLWEOp>>(
            &getContext());
    target.addLegalOp<mlir::tensor::GenerateOp, mlir::tensor::YieldOp>();

    // Add patterns to trivialy convert Concrete op to the equivalent
    // BConcrete op
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

    patterns.insert<GlweFromTablePattern>(&getContext());

    // Add patterns to rewrite tensor operators that works on encrypted tensors
    patterns
        .insert<ExtractSliceOpPattern, ExtractOpPattern, InsertSliceOpPattern,
                InsertOpPattern, FromElementsOpPattern>(&getContext());

    target.addDynamicallyLegalOp<mlir::tensor::ExtractSliceOp,
                                 mlir::tensor::ExtractOp, mlir::scf::YieldOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResultTypes()) &&
                 converter.isLegal(op->getOperandTypes());
        });

    patterns.insert<AllocTensorOpPattern>(&getContext());

    target.addDynamicallyLegalOp<mlir::tensor::InsertSliceOp,
                                 mlir::tensor::FromElementsOp,
                                 mlir::bufferization::AllocTensorOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResult(0).getType());
        });
    target.addLegalOp<mlir::memref::CopyOp>();

    patterns.insert<ForOpPattern>(&getContext());

    // Add patterns to rewrite some of memref ops that was introduced by the
    // linalg bufferization of encrypted tensor (first conversion of this pass)
    insertTensorShapeOpPattern<mlir::memref::ExpandShapeOp, mlir::MemRefType,
                               false>(getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::tensor::ExpandShapeOp, mlir::TensorType,
                               false>(getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::memref::CollapseShapeOp, mlir::MemRefType,
                               true>(getContext(), patterns, target);
    insertTensorShapeOpPattern<mlir::tensor::CollapseShapeOp, mlir::TensorType,
                               true>(getContext(), patterns, target);

    target.addDynamicallyLegalOp<
        mlir::arith::ConstantOp, mlir::scf::ForOp, mlir::scf::ParallelOp,
        mlir::scf::YieldOp, mlir::AffineApplyOp, mlir::memref::SubViewOp,
        mlir::memref::LoadOp, mlir::memref::TensorStoreOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResultTypes()) &&
                 converter.isLegal(op->getOperandTypes());
        });

    // Add patterns to do the conversion of func
    mlir::populateFunctionOpInterfaceTypeConversionPattern<mlir::func::FuncOp>(
        patterns, converter);

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return converter.isSignatureLegal(funcOp.getFunctionType()) &&
                 converter.isLegal(&funcOp.getBody());
        });

    target.addDynamicallyLegalOp<mlir::scf::ForOp>([&](mlir::scf::ForOp forOp) {
      return converter.isLegal(forOp.getInitArgs().getTypes()) &&
             converter.isLegal(forOp.getResults().getTypes());
    });

    // Add pattern for return op
    target.addDynamicallyLegalOp<mlir::func::ReturnOp>(
        [&](mlir::Operation *op) {
          return converter.isLegal(op->getResultTypes()) &&
                 converter.isLegal(op->getOperandTypes());
        });

    patterns.add<
        mlir::concretelang::GenericTypeConverterPattern<mlir::func::ReturnOp>,
        mlir::concretelang::GenericTypeConverterPattern<mlir::scf::YieldOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DataflowTaskOp>,
        mlir::concretelang::GenericTypeConverterPattern<
            mlir::concretelang::RT::DataflowYieldOp>>(&getContext(), converter);

    // Conversion of RT Dialect Ops
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowTaskOp>(target, converter);
    mlir::concretelang::addDynamicallyLegalTypeOp<
        mlir::concretelang::RT::DataflowYieldOp>(target, converter);

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
createConvertConcreteToBConcretePass(bool loopParallelize) {
  return std::make_unique<ConcreteToBConcretePass>(loopParallelize);
}
} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <unordered_set>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Support/Constants.h"

namespace arith = mlir::arith;
namespace linalg = mlir::linalg;
namespace tensor = mlir::tensor;
namespace bufferization = mlir::bufferization;

namespace FHE = mlir::concretelang::FHE;
namespace FHELinalg = mlir::concretelang::FHELinalg;

struct DotToLinalgGeneric
    : public ::mlir::OpRewritePattern<mlir::concretelang::FHELinalg::Dot> {
  DotToLinalgGeneric(::mlir::MLIRContext *context)
      : ::mlir::OpRewritePattern<::mlir::concretelang::FHELinalg::Dot>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  /// This rewrite pattern transforms any instance of
  /// `FHELinalg.dot_eint_int` to an instance of `linalg.generic` with an
  /// appropriate region using `FHE.mul_eint_int` and
  /// `FHE.add_eint` operations, an appropriate specification for the
  /// iteration dimensions and appropriate operations managing the
  /// accumulator of `linalg.generic`.
  ///
  /// Example:
  ///
  ///   %o = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
  ///     (tensor<4x!FHE.eint<0>>,
  ///      tensor<4xi32>) -> (!FHE.eint<0>)
  ///
  /// becomes:
  ///
  ///   %0 = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<0>>
  ///   %1 = linalg.generic {
  ///          indexing_maps = [#map0, #map0, #map1],
  ///          iterator_types = ["reduction"]
  ///        }
  ///        ins(%arg0, %arg1 : tensor<2x!FHE.eint<0>>, tensor<2xi32>)
  ///        outs(%0 : tensor<1x!FHE.eint<0>>) {
  ///          ^bb0(%arg2: !FHE.eint<0>, %arg3: i32, %arg4: !FHE.eint<0>):
  ///            %4 = "FHE.mul_eint_int"(%arg2, %arg3) :
  ///                    (!FHE.eint<0>, i32) -> !FHE.eint<0>
  ///
  ///            %5 = "FHE.add_eint"(%4, %arg4) :
  ///                    (!FHE.eint<0>, !FHE.eint<0>) -> !FHE.eint<0>
  ///
  ///            linalg.yield %5 : !FHE.eint<0>
  ///        } -> tensor<1x!FHE.eint<0>>
  ///
  ///   %c0 = constant 0 : index
  ///   %o = tensor.extract %1[%c0] : tensor<1x!FHE.eint<0>>
  ///
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::concretelang::FHELinalg::Dot dotOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto zeroTensorOp = rewriter.create<mlir::concretelang::FHE::ZeroTensorOp>(
        dotOp.getLoc(), mlir::RankedTensorType::get({1}, dotOp.getType()));

    // Create `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{zeroTensorOp.getType()};
    llvm::SmallVector<mlir::Value, 2> ins{dotOp.lhs(), dotOp.rhs()};
    llvm::SmallVector<mlir::Value, 1> outs{zeroTensorOp};
    llvm::SmallVector<mlir::AffineMap, 3> maps{
        mlir::AffineMap::getMultiDimIdentityMap(1, this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(1, this->getContext()),
        mlir::AffineMap::get(1, 0, {rewriter.getAffineConstantExpr(0)},
                             this->getContext())};

    llvm::SmallVector<llvm::StringRef, 1> itTypes{"reduction"};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    auto regBuilder = [&](mlir::OpBuilder &nestedBuilder,
                          mlir::Location nestedLoc,
                          mlir::ValueRange blockArgs) {
      mlir::concretelang::FHE::MulEintIntOp mul =
          nestedBuilder.create<mlir::concretelang::FHE::MulEintIntOp>(
              dotOp.getLoc(), blockArgs[0], blockArgs[1]);
      mlir::concretelang::FHE::AddEintOp add =
          nestedBuilder.create<mlir::concretelang::FHE::AddEintOp>(
              dotOp.getLoc(), mul, blockArgs[2]);

      nestedBuilder.create<mlir::linalg::YieldOp>(dotOp.getLoc(),
                                                  add.getResult());
    };

    mlir::linalg::GenericOp gop = rewriter.create<mlir::linalg::GenericOp>(
        dotOp.getLoc(), resTypes, ins, outs, maps, itTypes, doc, call,
        regBuilder);

    // Return value is still a 1-dimensional tensor; extract first
    // element and use it as a replacement for the result of the dot
    // operation
    mlir::Value idx0 =
        rewriter.create<mlir::arith::ConstantIndexOp>(dotOp.getLoc(), 0);
    llvm::SmallVector<mlir::Value, 1> indexes{idx0};
    mlir::Value res = rewriter.create<mlir::tensor::ExtractOp>(
        dotOp.getLoc(), gop.getResult(0), indexes);

    rewriter.replaceOp(dotOp, {res});

    return ::mlir::success();
  };
};

mlir::AffineMap
getBroadcastedAffineMap(const mlir::RankedTensorType &resultType,
                        const mlir::RankedTensorType &operandType,
                        ::mlir::PatternRewriter &rewriter) {
  mlir::SmallVector<mlir::AffineExpr, 4> affineExprs;
  auto resultShape = resultType.getShape();
  auto operandShape = operandType.getShape();
  affineExprs.reserve(operandShape.size());
  size_t deltaNumDim = resultShape.size() - operandShape.size();
  for (size_t i = 0; i < operandShape.size(); i++) {
    if (operandShape[i] == 1 && resultShape[i + deltaNumDim] != 1) {
      affineExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      affineExprs.push_back(rewriter.getAffineDimExpr(i + deltaNumDim));
    }
  }
  return mlir::AffineMap::get(resultShape.size(), 0, affineExprs,
                              rewriter.getContext());
}

/// This create an affine map following the broadcasting rules, but also takes
/// out one specific element of the LUT from the LUT dimension, which should be
/// the last.
///
/// Example:
///
/// resultType: 4x2x5, operandType: 4x2x8, lut_index: 3
/// return: affine_map<(d0, d1, d2) -> (d0, d1, 3)
/// last dimension of the operand is the lut size, and we take the map takes out
/// the element at index 3
mlir::AffineMap
getBroadcastedAffineMapMultiLUT(const mlir::RankedTensorType &resultType,
                                const mlir::RankedTensorType &operandType,
                                const int64_t lut_index,
                                ::mlir::PatternRewriter &rewriter) {
  mlir::SmallVector<mlir::AffineExpr, 4> affineExprs;
  auto resultShape = resultType.getShape();
  auto operandShape = operandType.getShape();
  affineExprs.reserve(operandShape.size());
  // Don't take the lut dimension into account
  size_t deltaNumDim = resultShape.size() - operandShape.size() + 1;
  for (size_t i = 0; i < operandShape.size() - 1; i++) {
    if (operandShape[i] == 1 && resultShape[i + deltaNumDim] != 1) {
      affineExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      affineExprs.push_back(rewriter.getAffineDimExpr(i + deltaNumDim));
    }
  }
  // Index a specific element of the LUT
  affineExprs.push_back(rewriter.getAffineConstantExpr(lut_index));
  return mlir::AffineMap::get(resultShape.size(), 0, affineExprs,
                              rewriter.getContext());
}

/// This template rewrite pattern transforms any instance of
/// operators `FHELinalgOp` that implements the broadasting rules to an
/// instance of `linalg.generic` with an appropriate region using `FHEOp`
/// operation, an appropriate specification for the iteration dimensions and
/// appropriate operations managing the accumulator of `linalg.generic`.
///
/// Example:
///
/// %res = FHELinalg.op(%lhs, %rhs):
/// (tensor<D$Ax...xD1x!FHE.eint<p>>, tensor<D$B'x...xD1'xT>)
///    -> tensor<DR"x...xD1"x!FHE.eint<p>>
///
/// becomes:
///
/// #maps_0 = [
///    affine_map<(a$R", ..., a$A, ..., a1) ->
///        (dim(lhs, $A) == 1 ? 0 : a$A,..., dim(lhs, 1) == 1 ? 0 : a1)>,
///    affine_map<(a$R", ..., a1) ->
///        (dim(rhs, $B') == 1 ? 0 : a$B', ..., dim(rhs, 1) == 1 ? 0 : a1)>,
///    affine_map<(a$R", ..., a1) -> (a$R", ..., a1)
/// ]
/// #attributes_0 {
///     indexing_maps = #maps_0,
///     iterator_types = ["parallel", ..., "parallel"], // $R" parallel
/// }
/// %init = linalg.init_tensor [DR",...,D1"]
///            : tensor<DR"x...xD1"x!FHE.eint<p>>
/// %res = linalg.generic {
///     ins(%lhs, %rhs: tensor<DAx...xD1x!FHE.eint<p>>,tensor<DB'x...xD1'xT>)
///     outs(%init : tensor<DR"x...xD1"x!FHE.eint<p>>)
///     {
///         ^bb0(%arg0: !FHE.eint<p>, %arg1: T):
///             %0 = FHE.op(%arg0, %arg1): !FHE.eint<p>, T ->
///             !FHE.eint<p>
///         linalg.yield %0 : !FHE.eint<p>
///     }
/// }
///
template <typename FHELinalgOp, typename FHEOp>
struct FHELinalgOpToLinalgGeneric : public mlir::OpRewritePattern<FHELinalgOp> {
  FHELinalgOpToLinalgGeneric(::mlir::MLIRContext *context,
                             mlir::PatternBenefit benefit =
                                 mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : ::mlir::OpRewritePattern<FHELinalgOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHELinalgOp linalgOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)linalgOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType lhsTy =
        ((mlir::Type)linalgOp.lhs().getType()).cast<mlir::RankedTensorType>();
    mlir::RankedTensorType rhsTy =
        ((mlir::Type)linalgOp.rhs().getType()).cast<mlir::RankedTensorType>();
    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        linalgOp.getLoc(), resultTy, mlir::ValueRange{});

    // Create the affine #maps_0
    llvm::SmallVector<mlir::AffineMap, 3> maps{
        getBroadcastedAffineMap(resultTy, lhsTy, rewriter),
        getBroadcastedAffineMap(resultTy, rhsTy, rewriter),
        getBroadcastedAffineMap(resultTy, resultTy, rewriter),
    };

    // Create the iterator_types
    llvm::SmallVector<llvm::StringRef> iteratorTypes(resultTy.getShape().size(),
                                                     "parallel");

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      FHEOp fheOp = nestedBuilder.create<FHEOp>(linalgOp.getLoc(), blockArgs[0],
                                                blockArgs[1]);

      nestedBuilder.create<mlir::linalg::YieldOp>(linalgOp.getLoc(),
                                                  fheOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 2> ins{linalgOp.lhs(), linalgOp.rhs()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(linalgOp.getLoc(), resTypes,
                                                 ins, outs, maps, iteratorTypes,
                                                 doc, call, bodyBuilder);

    rewriter.replaceOp(linalgOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

template <class T> inline mlir::RankedTensorType getRankedTensorType(T v) {
  return ((mlir::Type)v.getType()).cast<mlir::RankedTensorType>();
}

llvm::SmallVector<llvm::StringRef> parallelIteratorType(int n) {
  return llvm::SmallVector<llvm::StringRef>(n, "parallel");
}

/// This class rewrite pattern transforms any instance of
/// operators `FHELinalg.ApplyMappedLookupTableEintOp` that implements the
/// broadasting rules to an instance of `linalg.generic` with an appropriate
/// region using `FHE.ApplyLookupTableEintOp` operation, an appropriate
/// specification for the iteration dimensions and appropriate operations
/// managing the accumulator of `linalg.generic`.
///
/// The current implementation does not rely on 'tensor.extract_slice'
/// because of a bug in lowering this operation.
///
/// Example:
/// %res = "FHELinalg.apply_mapped_lookup_table"(%t, %luts, %map)
/// : (tensor<2x3x!FHE.eint<2>>, tensor<5x4xi64>, tensor<2x3xindex>)
/// -> tensor<2x3x!FHE.eint<2>>
///
/// becomes:
///
/// #map = affine_map<(d0, d1) -> (d0, d1)>
/// %init = linalg.init_tensor [2, 3] : tensor<2x3x!TFHE.glwe<{_,_,_}{2}>>
/// %output = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types
/// = ["parallel", "parallel"]} ins(%arg0, %arg2 :
/// tensor<2x3x!TFHE.glwe<{_,_,_}{2}>>, tensor<2x3xindex>) outs(%0 :
/// tensor<2x3x!TFHE.glwe<{_,_,_}{2}>>) {
///          ^bb0(%arg3: !TFHE.glwe<{_,_,_}{2}>, %lut_idx: index, %arg5:
///          !TFHE.glwe<{_,_,_}{2}>):  // no predecessors
///          // SHOULD BE
///          %lut = tensor.extract_slice %arg1[%[[LUTIDX]], 0] [1,4] [1, 1]
///                 : tensor<5x4xi64> to tensor<4xi64>
///          // BUT IS
///          %i0 = arith.constant 0 : index
///          ...
///          %i3 = arith.constant 3 : index
///          %e0 = tensor.extract %arg5[%lut_idx, %i0] : tensor<5x4xi64>
///          ...
///          %e3 = tensor.extract %arg5[%lut_idx, %i3] : tensor<5x4xi64>
///          %lut = tensor.from_elements %e0, ..., %e3 : tensor<4xi64>
///          %res  = "TFHE.apply_lookup_table"(%arg3, %[[LUT]])
///                    {baseLogBS = -1 : i32, baseLogKS = -1 : i32,
///                    glweDimension = -1 : i32,
///                      levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS =
///                      -1 : i32, polynomialSize = -1 : i32}
///                 : (!TFHE.glwe<{_,_,_}{2}>, tensor<4xi64>) ->
///          !TFHE.glwe<{_,_,_}{2}> linalg.yield %res :
///          !TFHE.glwe<{_,_,_}{2}>
/// } -> tensor<2x3x!TFHE.glwe<{_,_,_}{2}>>

namespace FHELinalg = mlir::concretelang::FHELinalg;

struct FHELinalgApplyMappedLookupTableToLinalgGeneric
    : public mlir::OpRewritePattern<FHELinalg::ApplyMappedLookupTableEintOp> {
  FHELinalgApplyMappedLookupTableToLinalgGeneric(
      ::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<FHELinalg::ApplyMappedLookupTableEintOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHELinalg::ApplyMappedLookupTableEintOp mappedLookup,
                  ::mlir::PatternRewriter &rewriter) const override {
    namespace arith = mlir::arith;
    namespace linalg = mlir::linalg;
    namespace tensor = mlir::tensor;
    namespace FHE = mlir::concretelang::FHE;
    using Values = llvm::SmallVector<mlir::Value>;
    using Types = llvm::SmallVector<mlir::Type>;
    using AffineMaps = llvm::SmallVector<mlir::AffineMap>;
    using sliceArg = llvm::SmallVector<mlir::OpFoldResult>;

    auto input = mappedLookup.t();
    auto luts = mappedLookup.luts();
    auto map = mappedLookup.map();

    auto loc = mappedLookup.getLoc();
    auto tensorTy = getRankedTensorType(input);
    auto lutsTy = getRankedTensorType(luts);
    auto resultTy = getRankedTensorType(mappedLookup->getResult(0));
    auto elementTy = resultTy.getElementType();
    auto lutElmtTy = lutsTy.getElementType();
    auto lutsShape = lutsTy.getShape();
    auto lutSize = lutsShape[lutsShape.size() - 1];
    auto resultShape = resultTy.getShape();

    auto integer = [&](auto v) -> mlir::Attribute {
      return rewriter.getI64IntegerAttr(v);
    };

    auto _0_ = integer(0);
    auto _1_ = integer(1);
    auto lutSizeValue = integer(lutSize);

    // Create the body of the `linalg.generic` op
    // %arg0 is an element of t (encrypted int)
    // %arg1 is an element of map (i64)
    // %arg2 is the output element
    auto lambdaBlock = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      auto tElmt = blockArgs[0];
      auto lutIdx = blockArgs[1];
      auto indexTy = rewriter.getIndexType();

      // %lut = extract_slice %luts[%lutIdx, 0][1, lutSize][1, 1]  :
      // tensor<NxKxi64> to tensor<Kxi64>
      mlir::Value lut;
      const bool WORKAROUND_EXTRACT_SLICE = true;
      if (!WORKAROUND_EXTRACT_SLICE) {
        sliceArg offsets{lutIdx, _0_};
        sliceArg sizes{_1_, lutSizeValue};
        sliceArg strides{_1_, _1_};
        auto lutTy = mlir::RankedTensorType::get(
            {static_cast<int64_t>(lutSize)}, lutElmtTy);
        lut = nestedBuilder.create<tensor::ExtractSliceOp>(
            loc, lutTy, luts, offsets, sizes, strides);
      } else {
        // WORKAROUND BEGIN
        // A bug in linalg-bufferize prevents rank reduction in extract_slice
        // Reshaping does not work either or is too complicated so let's rebuild
        // the tensor from scratch
        llvm::SmallVector<mlir::Value> consts;
        llvm::SmallVector<mlir::Value> extracts;
        for (int i = 0; i < lutSize; i++) {
          consts.push_back(
              // %5 = arith.constant(<i> : index) : index
              nestedBuilder.create<mlir::arith::ConstantOp>(
                  loc, indexTy, rewriter.getIndexAttr(i)));
        }
        for (int i = 0; i < lutSize; i++) {
          extracts.push_back(
              // %8 = tensor.extract %luts[<lutIdx>, <i>] : ...
              nestedBuilder.create<tensor::ExtractOp>(
                  loc, luts, mlir::ValueRange({lutIdx, consts[i]})));
        }
        // %12 = tensor.from_elements %8, ... : ...
        lut = nestedBuilder.create<tensor::FromElementsOp>(loc, extracts);
      } // WORKAROUND END
      // %res1 = apply_lookup_table %arg0 %lut
      auto lookup = nestedBuilder.create<FHE::ApplyLookupTableEintOp>(
          loc, elementTy, tElmt, lut);
      // linalg.yield %res1 : !FHE.eint<2>
      nestedBuilder.create<linalg::YieldOp>(loc, lookup.getResult());
    };

    auto output = rewriter.create<bufferization::AllocTensorOp>(
        loc, resultTy, mlir::ValueRange{});

    // Create the `linalg.g eneric` op
    Types resTys{resultTy};
    Values ins{input, map};
    Values outs{output};
    auto indexOfInput = getBroadcastedAffineMap(resultTy, tensorTy, rewriter);
    AffineMaps affineMaps{indexOfInput, indexOfInput, indexOfInput};
    auto iteratorTypes = parallelIteratorType(resultShape.size());
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resTys, ins, outs, affineMaps, iteratorTypes, lambdaBlock);

    rewriter.replaceOp(mappedLookup, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

/// This class rewrite pattern transforms any instance of
/// operators `FHELinalg.ApplyMultiLookupTableEintOp` that implements the
/// broadasting rules to an instance of `linalg.generic` with an appropriate
/// region using `FHE.ApplyLookupTableEintOp` operation, an appropriate
/// specification for the iteration dimensions and appropriate operaztions
/// managing the accumulator of `linalg.generic`.
///
/// Example:
///
/// %res = "FHELinalg.apply_multi_lookup_table"(%t, %luts):
/// (tensor<4x3x!FHE.eint<2>>, tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>>
///
/// becomes:
///
/// #maps_0 = [
///    affine_map<(d0, d1) -> (d0, d1)>
///    affine_map<(d0, d1) -> (d1, 0)>
///    affine_map<(d0, d1) -> (d1, 1)>
///    affine_map<(d0, d1) -> (d1, 2)>
///    affine_map<(d0, d1) -> (d1, 3)>
/// ]
/// #attributes_0 {
///     indexing_maps = #maps_0,
///     iterator_types = ["parallel", "parallel"],
/// }
/// %init = linalg.init_tensor [4, 3]
///            : tensor<4x3x!FHE.eint<2>>
/// %res = linalg.generic {
///     ins(%t, %luts, %luts, %luts, %luts: tensor<4x3x!FHE.eint<p>>,
///     tensor<3x4xi64>, tensor<3x4xi64>, tensor<3x4xi64>, tensor<3x4xi64>)
///     outs(%init : tensor<4x3x!FHE.eint<2>>)
///     {
///         ^bb0(%arg0: !FHE.eint<2>, %arg1: i64, %arg2: i64, %arg3: i64,
///         %arg4: i64, %arg5: !FHE.eint<2>):
///             %lut = tensor.from_elements %arg1, %arg2, %arg3, %arg4 :
///             tensor<4xi64> %0 = "TFHE.apply_lookup_table"(%arg0, %lut)
///             {baseLogBS = -1 : i32, baseLogKS = -1 : i32, glweDimension = -1
///             : i32, levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS = -1
///             : i32, polynomialSize = -1 : i32} : (!TFHE.glwe<{_,_,_}{2}>,
///             tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{2}>
///         linalg.yield %0 : !FHE.eint<2>
///     }
/// }
///
struct FHELinalgApplyMultiLookupTableToLinalgGeneric
    : public mlir::OpRewritePattern<
          mlir::concretelang::FHELinalg::ApplyMultiLookupTableEintOp> {
  FHELinalgApplyMultiLookupTableToLinalgGeneric(
      ::mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : ::mlir::OpRewritePattern<
            mlir::concretelang::FHELinalg::ApplyMultiLookupTableEintOp>(
            context, benefit) {}

  ::mlir::LogicalResult matchAndRewrite(
      mlir::concretelang::FHELinalg::ApplyMultiLookupTableEintOp fheLinalgLutOp,
      ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)fheLinalgLutOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType tensorTy = ((mlir::Type)fheLinalgLutOp.t().getType())
                                          .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType lutsTy =
        ((mlir::Type)fheLinalgLutOp.luts().getType())
            .cast<mlir::RankedTensorType>();
    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        fheLinalgLutOp.getLoc(), resultTy, mlir::ValueRange{});

    auto lutsShape = lutsTy.getShape();
    auto lut_size = lutsShape[lutsShape.size() - 1];
    // Create the affine maps
    llvm::SmallVector<mlir::AffineMap> maps{
        // Input tensor map
        getBroadcastedAffineMap(resultTy, tensorTy, rewriter)};
    maps.reserve(lut_size + 1);
    // Create as much affine maps as the size of the lut dimension
    for (int64_t i = 0; i < lut_size; i++)
      maps.push_back(
          getBroadcastedAffineMapMultiLUT(resultTy, lutsTy, i, rewriter));
    // Result map
    maps.push_back(getBroadcastedAffineMap(resultTy, resultTy, rewriter));

    // Create the iterator_types
    llvm::SmallVector<llvm::StringRef> iteratorTypes(resultTy.getShape().size(),
                                                     "parallel");

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      mlir::tensor::FromElementsOp lut =
          nestedBuilder.create<mlir::tensor::FromElementsOp>(
              fheLinalgLutOp.getLoc(), blockArgs.slice(1, lut_size));
      mlir::concretelang::FHE::ApplyLookupTableEintOp lutOp =
          nestedBuilder.create<mlir::concretelang::FHE::ApplyLookupTableEintOp>(
              fheLinalgLutOp.getLoc(), resultTy.getElementType(), blockArgs[0],
              lut.result());

      nestedBuilder.create<mlir::linalg::YieldOp>(fheLinalgLutOp.getLoc(),
                                                  lutOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value> ins{fheLinalgLutOp.t()};
    ins.reserve(lut_size + 2);
    // We extract one value at a time from one LUT using different maps, so we
    // need to pass the LUT `lut_size` time
    for (auto i = 0; i < lut_size; i++)
      ins.push_back(fheLinalgLutOp.luts());
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(
            fheLinalgLutOp.getLoc(), resTypes, ins, outs, maps, iteratorTypes,
            doc, call, bodyBuilder);

    rewriter.replaceOp(fheLinalgLutOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

/// This template rewrite pattern transforms any instance of
/// operators `FHELinalg.apply_lookup_table` that implements the broadasting
/// rules to an instance of `linalg.generic` with an appropriate region using
/// `FHE.apply_lookup_table` operation, an appropriate specification for the
/// iteration dimensions and appropriate operations managing the accumulator of
/// `linalg.generic`.
///
/// Example:
///
/// FHELinalg.apply_lookup_table(%t, %lut):
///  tensor<DNx...xD1x!FHE.eint<p>>, tensor<DAxi64>
///      -> tensor<DNx...xD1x!FHE.eint<p'>>
///
/// becomes:
///
/// #maps_0 = [
///    affine_map<(aN, ..., a1) -> (aN, ..., a1)>,
///    affine_map<(aN, ..., a1) -> (aN, ..., a1)>
/// ]
/// #attributes_0 {
///     indexing_maps = #maps_0,
///     iterator_types = ["parallel",..],//N parallel
/// }
/// %init = linalg.init_tensor [DN,...,D1]
///            : tensor<DNx...xD1x!FHE.eint<p'>>
/// %res = linalg.generic {
///     ins(%t: tensor<DNx...xD1x!FHE.eint<p>>)
///     outs(%init : tensor<DNx...xD1x!FHE.eint<p'>>)
///     {
///         ^bb0(%arg0: !FHE.eint<p>):
///             %0 = FHE.apply_lookup_table(%arg0, %lut): !FHE.eint<p>,
///             tensor<4xi64> -> !FHE.eint<p'>
///         linalg.yield %0 : !FHE.eint<p'>
///     }
/// }
///
struct FHELinalgApplyLookupTableToLinalgGeneric
    : public mlir::OpRewritePattern<
          mlir::concretelang::FHELinalg::ApplyLookupTableEintOp> {
  FHELinalgApplyLookupTableToLinalgGeneric(
      ::mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : ::mlir::OpRewritePattern<
            mlir::concretelang::FHELinalg::ApplyLookupTableEintOp>(context,
                                                                   benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::FHELinalg::ApplyLookupTableEintOp lutOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)lutOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType tTy =
        ((mlir::Type)lutOp.t().getType()).cast<mlir::RankedTensorType>();

    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        lutOp.getLoc(), resultTy, mlir::ValueRange{});

    // Create the affine #maps_0
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getMultiDimIdentityMap(tTy.getShape().size(),
                                                this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(resultTy.getShape().size(),
                                                this->getContext()),
    };

    // Create the iterator_types
    llvm::SmallVector<llvm::StringRef> iteratorTypes(resultTy.getShape().size(),
                                                     "parallel");

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      mlir::concretelang::FHE::ApplyLookupTableEintOp fheOp =
          nestedBuilder.create<mlir::concretelang::FHE::ApplyLookupTableEintOp>(
              lutOp.getLoc(), resultTy.getElementType(), blockArgs[0],
              lutOp.lut());

      nestedBuilder.create<mlir::linalg::YieldOp>(lutOp.getLoc(),
                                                  fheOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 1> ins{lutOp.t()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(lutOp.getLoc(), resTypes, ins,
                                                 outs, maps, iteratorTypes, doc,
                                                 call, bodyBuilder);

    rewriter.replaceOp(lutOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

/// This template rewrite pattern transforms any instance of
/// operators `FHELinalg.neg_eint` to an instance of `linalg.generic` with an
/// appropriate region using `FHE.neg_eint` operation, an appropriate
/// specification for the iteration dimensions and appropriate operations
/// managing the accumulator of `linalg.generic`.
///
/// Example:
///
/// FHELinalg.neg_eint(%tensor):
///  tensor<DNx...xD1x!FHE.eint<p>> -> tensor<DNx...xD1x!FHE.eint<p'>>
///
/// becomes:
///
/// #maps_0 = [
///    affine_map<(aN, ..., a1) -> (aN, ..., a1)>,
///    affine_map<(aN, ..., a1) -> (aN, ..., a1)>
/// ]
/// #attributes_0 {
///     indexing_maps = #maps_0,
///     iterator_types = ["parallel",..],//N parallel
/// }
/// %init = linalg.init_tensor [DN,...,D1]
///            : tensor<DNx...xD1x!FHE.eint<p'>>
/// %res = linalg.generic {
///     ins(%tensor: tensor<DNx...xD1x!FHE.eint<p>>)
///     outs(%init : tensor<DNx...xD1x!FHE.eint<p'>>)
///     {
///         ^bb0(%arg0: !FHE.eint<p>):
///             %0 = FHE.neg_eint(%arg0): !FHE.eint<p> -> !FHE.eint<p'>
///         linalg.yield %0 : !FHE.eint<p'>
///     }
/// }
///
struct FHELinalgNegEintToLinalgGeneric
    : public mlir::OpRewritePattern<mlir::concretelang::FHELinalg::NegEintOp> {
  FHELinalgNegEintToLinalgGeneric(
      ::mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : ::mlir::OpRewritePattern<mlir::concretelang::FHELinalg::NegEintOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::FHELinalg::NegEintOp negEintOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)negEintOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType tensorTy = ((mlir::Type)negEintOp.tensor().getType())
                                          .cast<mlir::RankedTensorType>();

    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<bufferization::AllocTensorOp>(
        negEintOp.getLoc(), resultTy, mlir::ValueRange{});

    // Create the affine #maps_0
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getMultiDimIdentityMap(tensorTy.getShape().size(),
                                                this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(resultTy.getShape().size(),
                                                this->getContext()),
    };

    // Create the iterator_types
    llvm::SmallVector<llvm::StringRef> iteratorTypes(resultTy.getShape().size(),
                                                     "parallel");

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      mlir::concretelang::FHE::NegEintOp fheOp =
          nestedBuilder.create<mlir::concretelang::FHE::NegEintOp>(
              negEintOp.getLoc(), resultTy.getElementType(), blockArgs[0]);

      nestedBuilder.create<mlir::linalg::YieldOp>(negEintOp.getLoc(),
                                                  fheOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 1> ins{negEintOp.tensor()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(negEintOp.getLoc(), resTypes,
                                                 ins, outs, maps, iteratorTypes,
                                                 doc, call, bodyBuilder);

    rewriter.replaceOp(negEintOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

/// This template rewrite pattern transforms any instance of
/// operators `FHELinalgMatmulOp` to an instance of `linalg.generic`
/// with an appropriate region using a builder that create the multiplication
/// operators and `FHE.add_eint` operation, an appropriate specification for
/// the iteration dimensions and appropriate operations managing the accumulator
/// of `linalg.generic`.
///
/// Example:
///
///  "FHELinalg.matmul_eint_int(%a, %b) :
///      (tensor<MxPx!FHE.eint<p>>, tensor<PxNxip'>) ->
///          tensor<MxNx!FHE.eint<p>>"
///
/// becomes:
///
/// #maps_0 = [
///   (m, n, p) -> (m, p),
///   (m, n, p) -> (p, n),
///   (m, n, p) -> (m, n)
/// ]
/// #attributes_0 = {
///   indexing_maps = #maps_0,
///   iterator_types = ["parallel", "parallel", "reduction"]
/// }
/// %init = FHE.zero_tensor : tensor<MxNx!FHE.eint<p>>
/// linalg.generic #attributes_0
///   ins(%A, %B : tensor<MxPx!FHE.eint<p>>,
///                tensor<PxNxip'>)
///   outs(%C : tensor<MxNx!FHE.eint<p>>)
///   {
///      ^bb0(%a: !FHE.eint<p>, %b: ip', %c: !FHE.eint<p>) :
///        %d = createMulOp(%a, %b): !FHE.eint<p>
///        %e = "FHE.add_eint"(%c, %d):
///              (!FHE.eint<p>, !FHE.eint<p>) -> !FHE.eint<p>
///        linalg.yield %e : !FHE.eint<p>
///   }
///
template <typename FHELinalgMatmulOp>
struct FHELinalgMatmulToLinalgGeneric
    : public mlir::OpRewritePattern<FHELinalgMatmulOp> {
  FHELinalgMatmulToLinalgGeneric(
      mlir::MLIRContext *context,
      std::function<FHE::MulEintIntOp(mlir::OpBuilder &, mlir::Location,
                                      mlir::Type, mlir::Value, mlir::Value)>
          createMulOp,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<FHELinalgMatmulOp>(context, benefit),
        createMulOp(createMulOp) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalgMatmulOp matmulOp,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Location location = matmulOp.getLoc();

    mlir::Value lhs = matmulOp.lhs();
    mlir::Value rhs = matmulOp.rhs();
    mlir::Value out = matmulOp.getResult();

    auto lhsType = ((mlir::Type)lhs.getType()).cast<mlir::RankedTensorType>();
    auto rhsType = ((mlir::Type)rhs.getType()).cast<mlir::RankedTensorType>();
    auto outType = ((mlir::Type)out.getType()).cast<mlir::RankedTensorType>();

    llvm::ArrayRef<int64_t> lhsShape = lhsType.getShape();
    llvm::ArrayRef<int64_t> rhsShape = rhsType.getShape();
    llvm::ArrayRef<int64_t> outShape = outType.getShape();

    int64_t lhsDims = (int64_t)lhsShape.size();
    int64_t rhsDims = (int64_t)rhsShape.size();
    int64_t outDims = (int64_t)outShape.size();

    mlir::Value zeros =
        rewriter.create<FHE::ZeroTensorOp>(location, outType).getResult();

    auto ins = llvm::SmallVector<mlir::Value, 2>{lhs, rhs};
    auto outs = llvm::SmallVector<mlir::Value, 1>{zeros};

    auto iteratorTypes = llvm::SmallVector<llvm::StringRef, 3>{};

    auto lhsAffineExpressions = llvm::SmallVector<mlir::AffineExpr, 2>{};
    auto rhsAffineExpressions = llvm::SmallVector<mlir::AffineExpr, 2>{};
    auto outAffineExpressions = llvm::SmallVector<mlir::AffineExpr, 2>{};

    if (lhsDims >= 2 && rhsDims >= 2) {

      // here are some example shapes to help understand the logic below
      // notation: lhs.shape @ rhs.shape -> output.shape

      // MxN @ NxP -> MxP

      // KxLxMxN @   NxP -> KxLxMxP
      // KxLxMxN @ LxNxP -> KxLxMxP
      // Kx1xMxN @ LxNxP -> KxLxMxP

      //   MxN @ KxLxNxP -> KxLxMxP
      // LxMxN @ KxLxNxP -> KxLxMxP
      // 1xMxN @ KxLxNxP -> KxLxMxP

      // make iterator types
      //   ["parallel", "parallel", ..., "parallel", "reduction"]
      //    ---------------------------------------
      //    output.shape.size() times
      //
      // think of it as
      //
      // - 1st iterator is for the 1st dimension of the output (K in examples)
      // - 2nd iterator is for the 2nd dimension of the output (L in examples)
      // - ...
      // - Nth iterator is for the Nth dimension of the output
      // - Last iterator is for the reduced dimension (N in the examples)

      for (int64_t i = 0; i < outDims; i++) {
        iteratorTypes.push_back(mlir::getParallelIteratorTypeName());
      }
      iteratorTypes.push_back(mlir::getReductionIteratorTypeName());

      // we need to put appropriate affine dimension expressions
      // that match lhs.shape on iterator types array

      // in KxLxMxN @ NxP -> KxLxMxP, we need to create the following map
      //
      // (dK, dL, dM, dP, dN) -> (dK, dL, dM, dN)
      // ==
      // (d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)

      // in LxMxN @ KxLxNxP -> KxLxMxP, we need to create the following map
      //
      // (dK, dL, dM, dP, dN) -> (dL, dM, dN)
      // ==
      // (d0, d1, d2, d3, d4) -> (d1, d2, d4)

      // in MxN @ KxLxNxP -> KxLxMxP, we need to create the following map
      //
      // (dK, dL, dM, dP, dN) -> (dM, dN)
      // ==
      // (d0, d1, d2, d3, d4) -> (d2, d4)

      // so the first AffineDimExpr we need to create is
      // output.shape.size() - lhs.shape.size() == outDims - lhsDims

      // then we need to add all dims in the output except it's last dim
      // so, we iterate up to output.shape.size() - 1 == outDims - 1

      // and finally, we add the AffineDimExpr corresponding to `N`
      // which is at the last index of `iteratorTypes`

      int64_t lhsDim = 0;
      for (int64_t outDim = outDims - lhsDims; outDim < outDims - 1; outDim++) {
        if (lhsDim < lhsDims - 2 && lhsShape[lhsDim] == 1) {
          // broadcasted so current `dim` will always be indexed with `0`
          lhsAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
        } else {
          assert(lhsShape[lhsDim] == outShape[outDim]);
          lhsAffineExpressions.push_back(rewriter.getAffineDimExpr(outDim));
        }
        lhsDim++;
      }
      lhsAffineExpressions.push_back(
          rewriter.getAffineDimExpr(iteratorTypes.size() - 1));

      // we need to put appropriate affine dimension expressions
      // that match rhs.shape on iterator types array

      // in KxLxMxN @ NxP -> KxLxMxP, we need to create the following map
      //
      // (dK, dL, dM, dP, dN) -> (dN, dP)
      // ==
      // (d0, d1, d2, d3, d4) -> (d4, d3)

      // in KxLxMxN @ LxNxP -> KxLxMxP, we need to create the following map
      //
      // (dK, dL, dM, dP, dN) -> (dL, dN, dP)
      // ==
      // (d0, d1, d2, d3, d4) -> (d1, d4, d3)

      // in LxMxN @ KxLxNxP -> KxLxMxP, we need to create the following map
      //
      // (dK, dL, dM, dP, dN) -> (dK, dL, dN, dP)
      // ==
      // (d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)

      // so the first AffineDimExpr we need to create is
      // output.shape.size() - rhs.shape.size() == outDims - rhsDims

      // then we need to add all dims in the output except it's last 2 dims
      // so, we iterate up to output.shape.size() - 2 == outDims - 2

      // and finally, we add the AffineDimExpr corresponding to `N` and `P`
      // which is at the last and one before last indices of `iteratorTypes`

      int64_t rhsDim = 0;
      for (int64_t outDim = outDims - rhsDims; outDim < outDims - 2; outDim++) {
        if (rhsShape[rhsDim] == 1) {
          // broadcasted so current `dim` will always be indexed with `0`
          rhsAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
        } else {
          assert(rhsShape[rhsDim] == outShape[outDim]);
          rhsAffineExpressions.push_back(rewriter.getAffineDimExpr(outDim));
        }
        rhsDim++;
      }
      rhsAffineExpressions.push_back(
          rewriter.getAffineDimExpr(iteratorTypes.size() - 1));
      rhsAffineExpressions.push_back(
          rewriter.getAffineDimExpr(iteratorTypes.size() - 2));

      for (int64_t i = 0; i < outDims; i++) {
        outAffineExpressions.push_back(rewriter.getAffineDimExpr(i));
      }

    } else if (lhsDims == 1 && rhsDims >= 2) {

      // here are some example shapes to help understand the logic below
      // notation: lhs.shape @ rhs.shape -> output.shape

      // N @     NxP ->     P
      // N @   LxNxP ->   LxP
      // N @ KxLxNxP -> KxLxP

      int64_t commonDim = rhsDims - 2;
      for (int64_t i = 0; i < rhsDims; i++) {
        if (i == commonDim) {
          iteratorTypes.push_back(mlir::getReductionIteratorTypeName());
        } else {
          iteratorTypes.push_back(mlir::getParallelIteratorTypeName());
        }
      }

      lhsAffineExpressions.push_back(rewriter.getAffineDimExpr(commonDim));

      for (int64_t i = 0; i < rhsDims; i++) {
        rhsAffineExpressions.push_back(rewriter.getAffineDimExpr(i));
      }

      for (int64_t i = 0; i < rhsDims; i++) {
        if (i != commonDim) {
          outAffineExpressions.push_back(rewriter.getAffineDimExpr(i));
        }
      }

    } else if (lhsDims >= 2 && rhsDims == 1) {

      // here are some example shapes to help understand the logic below
      // notation: lhs.shape @ rhs.shape -> output.shape

      //     MxN @ N ->     M
      //   LxMxN @ N ->   LxM
      // KxLxMxN @ N -> KxLxM

      for (int64_t i = 0; i < lhsDims - 1; i++) {
        iteratorTypes.push_back(mlir::getParallelIteratorTypeName());
      }
      iteratorTypes.push_back(mlir::getReductionIteratorTypeName());

      for (int64_t i = 0; i < lhsDims; i++) {
        lhsAffineExpressions.push_back(rewriter.getAffineDimExpr(i));
      }

      rhsAffineExpressions.push_back(rewriter.getAffineDimExpr(lhsDims - 1));

      for (int64_t i = 0; i < lhsDims - 1; i++) {
        outAffineExpressions.push_back(rewriter.getAffineDimExpr(i));
      }
    }

    auto maps = llvm::SmallVector<mlir::AffineMap, 3>{
        mlir::AffineMap::get(iteratorTypes.size(), 0, lhsAffineExpressions,
                             rewriter.getContext()),
        mlir::AffineMap::get(iteratorTypes.size(), 0, rhsAffineExpressions,
                             rewriter.getContext()),
        mlir::AffineMap::get(iteratorTypes.size(), 0, outAffineExpressions,
                             rewriter.getContext()),
    };

    mlir::Type outElementType = outType.getElementType();
    auto regionBuilder = [&](mlir::OpBuilder &nestedBuilder,
                             mlir::Location nestedLoc,
                             mlir::ValueRange blockArgs) {
      mlir::Value multiplication =
          createMulOp(nestedBuilder, location, outElementType, blockArgs[0],
                      blockArgs[1])
              .getResult();

      mlir::Value addition =
          nestedBuilder
              .create<FHE::AddEintOp>(location, outElementType, blockArgs[2],
                                      multiplication)
              .getResult();

      nestedBuilder.create<linalg::YieldOp>(location, addition);
    };

    auto resultTypes = llvm::SmallVector<mlir::Type, 1>{outType};
    mlir::Value result =
        rewriter
            .create<linalg::GenericOp>(location, resultTypes, ins, outs, maps,
                                       iteratorTypes, regionBuilder)
            .getResult(0);

    rewriter.replaceOp(matmulOp, {result});
    return mlir::success();
  };

private:
  std::function<FHE::MulEintIntOp(mlir::OpBuilder &, mlir::Location, mlir::Type,
                                  mlir::Value, mlir::Value)>
      createMulOp;
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.sum` to an instance of `linalg.generic`.
///
/// Example:
///
///   %result = "FHELinalg.sum"(%input) :
///     tensor<d0xd1x...xdNx!FHE.eint<p>>() -> !FHE.eint<p>
///
/// becomes:
///
///   #map0 = affine_map<(i0, i1, ..., iN) -> (i0, i1, ..., iN)>
///   #map1 = affine_map<(i0, i1, ..., iN) -> (0)>
///
///   %accumulator = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
///   %accumulation = linalg.generic
///     {
///       indexing_maps = [#map0, #map1],
///       iterator_types = ["reduction", "reduction", ..., "reduction"]
///     }
///     ins(%input : tensor<d0xd1x...xdNx!FHE.eint<7>>)
///     outs(%accumulator : tensor<1x!FHE.eint<7>>)
///     {
///       ^bb0(%a: !FHE.eint<7>, %b: !FHE.eint<7>):
///         %c = "FHE.add_eint"(%a, %b) :
///           (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
///         linalg.yield %c : !FHE.eint<7>
///     } -> tensor<1x!FHE.eint<7>>
///
///   %index = arith.constant 0 : index
///   %result = tensor.extract %index : tensor<1x!FHE.eint<7>>
///
struct SumToLinalgGeneric
    : public ::mlir::OpRewritePattern<mlir::concretelang::FHELinalg::SumOp> {
  SumToLinalgGeneric(::mlir::MLIRContext *context)
      : ::mlir::OpRewritePattern<::mlir::concretelang::FHELinalg::SumOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::concretelang::FHELinalg::SumOp sumOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    mlir::Location location = sumOp.getLoc();

    mlir::Value input = sumOp.getOperand();
    mlir::Value output = sumOp.getResult();

    auto inputType = input.getType().dyn_cast<mlir::TensorType>();
    mlir::Type outputType = output.getType();

    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t inputDimensions = inputShape.size();

    bool outputIsTensor = outputType.isa<mlir::TensorType>();

    for (int64_t size : inputShape) {
      if (size == 0) {
        mlir::Value result;
        if (outputIsTensor) {
          result = rewriter.create<FHE::ZeroTensorOp>(location, outputType)
                       .getResult();
        } else {
          result = rewriter.create<FHE::ZeroEintOp>(location, outputType)
                       .getResult();
        }
        rewriter.replaceOp(sumOp, {result});
        return mlir::success();
      }
    }

    auto axesToDestroy = std::unordered_set<int64_t>{};
    for (mlir::Attribute axisAttribute : sumOp.axes()) {
      int64_t axis = axisAttribute.cast<mlir::IntegerAttr>().getInt();
      axesToDestroy.insert(axis);
    }
    if (axesToDestroy.empty()) {
      for (int64_t i = 0; i < inputDimensions; i++) {
        axesToDestroy.insert(i);
      }
    }

    mlir::Type accumulatorType = outputType;
    if (!outputIsTensor) {
      int64_t accumulatorShape[1] = {1};
      accumulatorType = // tensor of shape (1,)
          mlir::RankedTensorType::get(accumulatorShape, outputType);
    }

    mlir::Value accumulator =
        rewriter.create<FHE::ZeroTensorOp>(location, accumulatorType)
            .getResult();

    auto ins = llvm::SmallVector<mlir::Value, 1>{input};
    auto outs = llvm::SmallVector<mlir::Value, 1>{accumulator};

    mlir::AffineMap inputMap = mlir::AffineMap::getMultiDimIdentityMap(
        inputDimensions, this->getContext());

    auto outputAffineExpressions = llvm::SmallVector<mlir::AffineExpr, 3>{};
    if (outputIsTensor) {
      for (int64_t i = 0; i < inputDimensions; i++) {
        bool ithAxisIsDestroyed = axesToDestroy.find(i) != axesToDestroy.end();
        if (!ithAxisIsDestroyed) {
          outputAffineExpressions.push_back(rewriter.getAffineDimExpr(i));
        } else if (sumOp.keep_dims()) {
          outputAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
        }
      }
    } else {
      outputAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
    }

    mlir::AffineMap outputMap = mlir::AffineMap::get(
        inputDimensions, 0, outputAffineExpressions, rewriter.getContext());

    auto maps = llvm::SmallVector<mlir::AffineMap, 2>{inputMap, outputMap};

    auto iteratorTypes = llvm::SmallVector<llvm::StringRef, 3>(
        inputDimensions, mlir::getParallelIteratorTypeName());

    for (int64_t axis : axesToDestroy) {
      iteratorTypes[axis] = mlir::getReductionIteratorTypeName();
    }

    auto regionBuilder = [&](mlir::OpBuilder &nestedBuilder,
                             mlir::Location nestedLoc,
                             mlir::ValueRange blockArgs) {
      mlir::Value lhs = blockArgs[0];
      mlir::Value rhs = blockArgs[1];
      mlir::Value addition =
          nestedBuilder.create<FHE::AddEintOp>(location, lhs, rhs).getResult();

      nestedBuilder.create<linalg::YieldOp>(location, addition);
    };

    auto resultTypes = llvm::SmallVector<mlir::Type, 1>{accumulatorType};
    mlir::Value accumulation =
        rewriter
            .create<linalg::GenericOp>(location, resultTypes, ins, outs, maps,
                                       iteratorTypes, regionBuilder)
            .getResult(0);

    mlir::Value result = accumulation;
    if (!outputIsTensor) {
      auto indices = llvm::SmallVector<mlir::Value, 1>{
          rewriter.create<arith::ConstantIndexOp>(location, 0).getResult(),
      };
      result =
          rewriter.create<tensor::ExtractOp>(location, accumulation, indices)
              .getResult();
    }

    rewriter.replaceOp(sumOp, {result});
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.transpose` to an instance of `linalg.generic`.
///
/// Example:
///
///   %result = "FHELinalg.transpose"(%input: tensor<d0xd1x...xdNx!FHE.eint<p>>)
///   -> tensor<dNx...xd1xd0x!FHE.eint<p>
///
/// becomes:
///
///   #map0 = affine_map<(i0, i1, ..., iN) -> (iN, ..., i1, i0)>
///   #map1 = affine_map<(i0, i1, ..., iN) -> (i0, i1, ..., iN)>
///
///   %accumulator = "FHE.zero_tensor"() : () ->
///   tensor<dNx...xd1xd0x!FHE.eint<6>> %result = linalg.generic
///     {
///       indexing_maps = [#map0, #map1],
///       iterator_types = ["parallel", "parallel", ..., "parallel"]
///     }
///     ins(%input : tensor<d0xd1x...xdNx!FHE.eint<7>>)
///     outs(%accumulator : tensor<dNx...xd1xd0x!FHE.eint<7>>)
///     {
///       ^bb0(%a: !FHE.eint<7>, %b: !FHE.eint<7>):
///         linalg.yield %a : !FHE.eint<7>
///     } -> tensor<dNx...xd1xd0x!FHE.eint<7>>
///
struct TransposeToLinalgGeneric
    : public ::mlir::OpRewritePattern<
          mlir::concretelang::FHELinalg::TransposeOp> {
  TransposeToLinalgGeneric(::mlir::MLIRContext *context)
      : ::mlir::OpRewritePattern<::mlir::concretelang::FHELinalg::TransposeOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::concretelang::FHELinalg::TransposeOp transposeOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    mlir::Value input = transposeOp.getOperand();
    mlir::Value output = transposeOp.getResult();
    auto inputType = input.getType().dyn_cast<mlir::RankedTensorType>();
    auto outputType = output.getType().dyn_cast<mlir::RankedTensorType>();

    mlir::Location location = transposeOp.getLoc();
    // Initialize empty tensor to fill with transpose result
    mlir::Value zeroTensor =
        rewriter.create<FHE::ZeroTensorOp>(location, outputType).getResult();

    // Inverted dimensions to create a transposition
    std::vector<unsigned int> perms = {};
    auto n_dim = inputType.getShape().size();
    for (int i = n_dim - 1; i >= 0; i--)
      perms.push_back(i);

    llvm::SmallVector<mlir::Type, 1> resultTypes{zeroTensor.getType()};
    auto ins = llvm::SmallVector<mlir::Value, 1>{input};
    auto outs = llvm::SmallVector<mlir::Value, 1>{zeroTensor};
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getPermutationMap(perms, this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(n_dim, this->getContext()),
    };
    auto iteratorTypes = parallelIteratorType(n_dim);
    // The maps will be responsible for changing item positions, we just return
    // items here
    auto regionBuilder = [&](mlir::OpBuilder &nestedBuilder,
                             mlir::Location nestedLoc,
                             mlir::ValueRange blockArgs) {
      mlir::Value item = blockArgs[0];
      nestedBuilder.create<linalg::YieldOp>(location, item);
    };
    mlir::Value result =
        rewriter
            .create<linalg::GenericOp>(location, resultTypes, ins, outs, maps,
                                       iteratorTypes, regionBuilder)
            .getResult(0);

    rewriter.replaceOp(transposeOp, {result});
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.from_element` to an instance of `tensor.from_elements`.
///
/// Example:
///
///   %result = "FHELinalg.from_element"(%x) : (Type) -> tensor<1xType>
///
/// becomes:
///
///   %result = tensor.from_elements %x : (Type) -> tensor<1xType>
///
struct FromElementToTensorFromElements
    : public ::mlir::OpRewritePattern<
          mlir::concretelang::FHELinalg::FromElementOp> {

  FromElementToTensorFromElements(::mlir::MLIRContext *context)
      : ::mlir::OpRewritePattern<
            ::mlir::concretelang::FHELinalg::FromElementOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::concretelang::FHELinalg::FromElementOp op,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto in = op.getOperand();
    auto out = op.getResult();

    mlir::Value result =
        rewriter.create<tensor::FromElementsOp>(op.getLoc(), out.getType(), in)
            .getResult();

    rewriter.replaceOp(op, {result});
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.concat` to instances of `tensor.insert_slice`
///
/// Example:
///
///   %result = "FHELinalg.concat"(%x, %y) { axis = 1 } :
///     (tensor<2x3x!FHE.eint<4>>, tensor<2x4x!FHE.eint<4>>)
///       -> tensor<2x7x!FHE.eint<4>>
///
/// becomes:
///
///   %empty = "FHE.zero_tensor"() : () -> tensor<2x7x!FHE.eint<4>>
///
///   %x_copied = tensor.insert_slice %x into %empty[0, 0] [2, 3] [1, 1]
///     : tensor<2x3x!FHE.eint<4>> into tensor<2x7x!FHE.eint<4>>
///
///   %y_copied = tensor.insert_slice %y into %x_copied[0, 3] [2, 4] [1, 1]
///     : tensor<2x4x!FHE.eint<4>> into tensor<2x7x!FHE.eint<4>>
///
struct ConcatRewritePattern
    : public mlir::OpRewritePattern<FHELinalg::ConcatOp> {
  ConcatRewritePattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHELinalg::ConcatOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalg::ConcatOp op,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Location location = op.getLoc();
    size_t axis = op.axis();

    mlir::Value output = op.getResult();
    auto outputType = output.getType().dyn_cast<mlir::TensorType>();

    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
    size_t outputDimensions = outputShape.size();

    mlir::Value result =
        rewriter.create<FHE::ZeroTensorOp>(location, outputType).getResult();

    auto offsets = llvm::SmallVector<int64_t, 3>{};
    auto sizes = llvm::SmallVector<int64_t, 3>{};
    auto strides = llvm::SmallVector<int64_t, 3>{};

    // set up the initial values of offsets, sizes, and strides
    // each one has exactly `outputDimensions` number of elements
    // - offsets will be [0, 0, 0, ..., 0, 0, 0]
    // - strides will be [1, 1, 1, ..., 1, 1, 1]
    // - sizes will be the output shape except at the 'axis' which will be 0
    for (size_t i = 0; i < outputDimensions; i++) {
      offsets.push_back(0);
      if (i == axis) {
        sizes.push_back(0);
      } else {
        sizes.push_back(outputShape[i]);
      }
      strides.push_back(1);
    }

    // these are not used, but they are required
    // for the creation of InsertSliceOp operation
    auto dynamicOffsets = llvm::ArrayRef<mlir::Value>{};
    auto dynamicSizes = llvm::ArrayRef<mlir::Value>{};
    auto dynamicStrides = llvm::ArrayRef<mlir::Value>{};

    for (mlir::Value input : op.getOperands()) {
      auto inputType = input.getType().dyn_cast<mlir::TensorType>();
      int64_t axisSize = inputType.getShape()[axis];

      // offsets and sizes will be modified for each input tensor
      // if we have:
      //     "FHELinalg.concat"(%x, %y, %z) :
      //     (
      //         tensor<3x!FHE.eint<7>>,
      //         tensor<4x!FHE.eint<7>>,
      //         tensor<2x!FHE.eint<7>>,
      //     )
      //     -> tensor<9x!FHE.eint<7>>
      //
      // for the first copy:
      //     offsets = [0], sizes = [3], strides = [1]
      //
      // for the second copy:
      //     offsets = [3], sizes = [4], strides = [1]
      //
      // for the third copy:
      //     offsets = [7], sizes = [2], strides = [1]
      //
      // so in each iteration:
      // - the size is set to the axis size of the input
      // - the offset is increased by the size of the previous input

      sizes[axis] = axisSize;

      // these arrays are copied, so it's fine to modify and use them again
      mlir::ArrayAttr offsetsAttr = rewriter.getI64ArrayAttr(offsets);
      mlir::ArrayAttr sizesAttr = rewriter.getI64ArrayAttr(sizes);
      mlir::ArrayAttr stridesAttr = rewriter.getI64ArrayAttr(strides);

      offsets[axis] += axisSize;

      result = rewriter
                   .create<mlir::tensor::InsertSliceOp>(
                       location, outputType, input, result, dynamicOffsets,
                       dynamicSizes, dynamicStrides, offsetsAttr, sizesAttr,
                       stridesAttr)
                   .getResult();
    }

    rewriter.replaceOp(op, {result});
    return mlir::success();
  };
};

static mlir::SmallVector<mlir::OpFoldResult>
getAsOpFoldResult(mlir::OpBuilder &b, mlir::Location loc,
                  mlir::SmallVectorImpl<int64_t> &ints) {
  return llvm::to_vector<4>(
      llvm::map_range(ints, [&](int64_t val) -> mlir::OpFoldResult {
        return b.getIndexAttr(val);
      }));
}

/// Helper function to get the padding tensor given the padding int values, and
/// the value to pad with
static mlir::Value
getPaddedTensor(mlir::Operation *op, mlir::OpBuilder &b, mlir::Value &input,
                mlir::SmallVectorImpl<int64_t> &lowPaddingInts,
                mlir::SmallVectorImpl<int64_t> &highPaddingInts,
                mlir::Value pad) {
  assert(input.getType().isa<mlir::RankedTensorType>() &&
         "input must be RankedTensorType");
  mlir::Location loc = op->getLoc();
  mlir::Type rankedTensorType = mlir::tensor::PadOp::inferResultType(
      input.getType().cast<mlir::RankedTensorType>(), lowPaddingInts,
      highPaddingInts);
  mlir::SmallVector<mlir::OpFoldResult> lowPaddings =
      getAsOpFoldResult(b, loc, lowPaddingInts);
  mlir::SmallVector<mlir::OpFoldResult> highPaddings =
      getAsOpFoldResult(b, loc, highPaddingInts);
  mlir::Value paddedInput = mlir::tensor::createPadScalarOp(
      rankedTensorType, input, pad, /*low=*/lowPaddings, /*high=*/highPaddings,
      /*packing=*/false, loc, b);
  return paddedInput;
}

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.conv2d` to an instance of `linalg.fhelinalg_conv_2d_nchw_fchw`.
/// The transformation consists of padding the input tensor, and initializing
/// the output tensor with bias values if any.
struct FHELinalgConv2dToLinalgConv2d
    : public ::mlir::OpRewritePattern<mlir::concretelang::FHELinalg::Conv2dOp> {
  FHELinalgConv2dToLinalgConv2d(::mlir::MLIRContext *context)
      : ::mlir::OpRewritePattern<::mlir::concretelang::FHELinalg::Conv2dOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::concretelang::FHELinalg::Conv2dOp conv2dOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    mlir::Location loc = conv2dOp->getLoc();
    mlir::Value input =
        conv2dOp.input(); /* shape: Batch*Channels*Height*Width */
    mlir::Value weight =
        conv2dOp.weight(); /* shape: Filters*Channels*Height*Width */

    mlir::Type inputElementType =
        input.getType().cast<mlir::RankedTensorType>().getElementType();

    // Attriutes are assumed to be correct after passing the verification
    mlir::SmallVector<int64_t, 4> paddingInts =
        mlir::concretelang::FHELinalg::getPaddingFromConv2d(conv2dOp);
    mlir::SmallVector<int64_t, 2> stridesInts =
        mlir::concretelang::FHELinalg::getStridesFromConv2d(conv2dOp);
    mlir::SmallVector<int64_t, 2> dilationsInts =
        mlir::concretelang::FHELinalg::getDilationsFromConv2d(conv2dOp);

    // Pad the input tensor according to padding.
    mlir::SmallVector<int64_t, 4> lowPaddingIncludingNC = {0, 0};
    lowPaddingIncludingNC.insert(lowPaddingIncludingNC.end(),
                                 paddingInts.begin() + 2, paddingInts.end());
    mlir::SmallVector<int64_t, 4> highPaddingIncludingNC = {0, 0};
    highPaddingIncludingNC.insert(highPaddingIncludingNC.end(),
                                  paddingInts.begin(), paddingInts.begin() + 2);
    mlir::Value paddingValue =
        rewriter.create<mlir::concretelang::FHE::ZeroEintOp>(
            loc,
            input.getType().cast<mlir::RankedTensorType>().getElementType());
    mlir::Value paddedInput =
        getPaddedTensor(conv2dOp, rewriter, input, lowPaddingIncludingNC,
                        highPaddingIncludingNC, paddingValue);

    // TODO(Optimization): output tensor is being constructed in two different
    // ways, depending of whether there is a bias or not:
    // 1- There is no bias: we initialize the output tensor to encryptions of
    // zero
    // 2- There is a bias: we initialize the output tensor to encryptions of
    // zeros, then we add bias values.
    // For the second case, it can be done by initializing the output to
    // encryption of bias values directly
    mlir::Value initTensor =
        rewriter.create<mlir::concretelang::FHE::ZeroTensorOp>(
            loc, mlir::RankedTensorType::get(conv2dOp.getResult()
                                                 .getType()
                                                 .cast<mlir::RankedTensorType>()
                                                 .getShape(),
                                             inputElementType));
    // Since linalg doesn't support a bias in the conv operation, we initialize
    // the output tensor to the bias values, so that conv results get
    // accumulated to it
    mlir::Value bias = conv2dOp.bias(); /* optional of shape: Filters */
    mlir::Value biasInitTensor;
    if (!bias) { // no bias was used
      biasInitTensor = initTensor;
    } else {
      // Fill the output tensor with bias values
      auto resultRank =
          initTensor.getType().cast<mlir::RankedTensorType>().getRank();
      mlir::SmallVector<mlir::AffineMap> indexingMaps = {
          mlir::AffineMap::get(resultRank, 0, rewriter.getAffineDimExpr(1),
                               rewriter.getContext()),
          rewriter.getMultiDimIdentityMap(resultRank)};
      mlir::SmallVector<llvm::StringRef> iteratorTypes(resultRank, "parallel");
      biasInitTensor =
          rewriter
              .create<mlir::linalg::GenericOp>(
                  loc, initTensor.getType(), bias, initTensor, indexingMaps,
                  iteratorTypes,
                  [](mlir::OpBuilder &b, mlir::Location loc,
                     mlir::ValueRange args) {
                    mlir::Value encryptedBias =
                        b.create<mlir::concretelang::FHE::AddEintIntOp>(
                             loc, args[1], args[0])
                            .getResult();
                    b.create<mlir::linalg::YieldOp>(loc, encryptedBias);
                  })
              .getResult(0);
    }

    auto stridesAttr = rewriter.getI64VectorAttr(stridesInts);
    auto dilationsAttr = rewriter.getI64VectorAttr(dilationsInts);
    rewriter.replaceOpWithNewOp<
        mlir::concretelang::FHELinalg::FhelinalgConv2DNchwFchwOp>(
        conv2dOp, biasInitTensor.getType(),
        mlir::ValueRange{paddedInput, weight}, biasInitTensor, stridesAttr,
        dilationsAttr);
    return mlir::success();
  };
};

namespace {
struct FHETensorOpsToLinalg
    : public FHETensorOpsToLinalgBase<FHETensorOpsToLinalg> {

  void runOnOperation() final;
};

void FHETensorOpsToLinalg::runOnOperation() {
  mlir::func::FuncOp function = this->getOperation();

  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::concretelang::FHE::FHEDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
  target.addLegalDialect<mlir::arith::ArithmeticDialect>();
  target.addIllegalOp<mlir::concretelang::FHELinalg::Dot>();
  target.addIllegalDialect<mlir::concretelang::FHELinalg::FHELinalgDialect>();
  // TODO: this should be removed when we no longer need a custom generated op
  // for conv that works on tensors of custom types
  target.addLegalOp<mlir::concretelang::FHELinalg::FhelinalgConv2DNchwFchwOp>();

  target.addLegalOp<bufferization::AllocTensorOp>();

  mlir::RewritePatternSet patterns(&getContext());
  patterns.insert<DotToLinalgGeneric>(&getContext());
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::AddEintOp,
                                 mlir::concretelang::FHE::AddEintOp>>(
      &getContext());
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::AddEintIntOp,
                                 mlir::concretelang::FHE::AddEintIntOp>>(
      &getContext());
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::SubIntEintOp,
                                 mlir::concretelang::FHE::SubIntEintOp>>(
      &getContext());
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::SubEintIntOp,
                                 mlir::concretelang::FHE::SubEintIntOp>>(
      &getContext());
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::SubEintOp,
                                 mlir::concretelang::FHE::SubEintOp>>(
      &getContext());
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::MulEintIntOp,
                                 mlir::concretelang::FHE::MulEintIntOp>>(
      &getContext());
  patterns.insert<FHELinalgApplyLookupTableToLinalgGeneric>(&getContext());
  patterns.insert<FHELinalgNegEintToLinalgGeneric>(&getContext());
  patterns.insert<FHELinalgMatmulToLinalgGeneric<
      mlir::concretelang::FHELinalg::MatMulEintIntOp>>(
      &getContext(), [](mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Type type, mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintIntOp>(
            loc, type, arg0, arg1);
      });
  patterns.insert<FHELinalgMatmulToLinalgGeneric<
      mlir::concretelang::FHELinalg::MatMulIntEintOp>>(
      &getContext(), [](mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Type type, mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintIntOp>(
            loc, type, arg1, arg0);
      });
  patterns.insert<FHELinalgApplyMultiLookupTableToLinalgGeneric>(&getContext());
  patterns.insert<FHELinalgApplyMappedLookupTableToLinalgGeneric>(
      &getContext());
  patterns.insert<SumToLinalgGeneric>(&getContext());
  patterns.insert<ConcatRewritePattern>(&getContext());
  patterns.insert<FHELinalgConv2dToLinalgConv2d>(&getContext());
  patterns.insert<TransposeToLinalgGeneric>(&getContext());
  patterns.insert<FromElementToTensorFromElements>(&getContext());

  if (mlir::applyPartialConversion(function, target, std::move(patterns))
          .failed())
    this->signalPassFailure();
}

} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createConvertFHETensorOpsToLinalg() {
  return std::make_unique<FHETensorOpsToLinalg>();
}
} // namespace concretelang
} // namespace mlir

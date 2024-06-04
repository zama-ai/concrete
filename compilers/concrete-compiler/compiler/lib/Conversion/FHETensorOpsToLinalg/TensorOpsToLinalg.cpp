// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Dialect/Optimizer/IR/OptimizerOps.h"
#include "concretelang/Support/Constants.h"
#include "concretelang/Support/logging.h"

#include <unordered_set>

namespace arith = mlir::arith;
namespace linalg = mlir::linalg;
namespace tensor = mlir::tensor;
namespace bufferization = mlir::bufferization;

namespace FHE = mlir::concretelang::FHE;
namespace FHELinalg = mlir::concretelang::FHELinalg;

inline void forwardOptimizerID(mlir::Operation *source,
                               mlir::Operation *destination) {
  auto optimizerIdAttr = source->getAttr("TFHE.OId");
  if (optimizerIdAttr == nullptr) {
    mlir::concretelang::log_verbose() << "No TFHE.OId\n";
    return;
  }
  destination->setAttr("TFHE.OId", optimizerIdAttr);
}

template <typename DotOp, typename FHEMulOp>
struct DotToLinalgGeneric : public ::mlir::OpRewritePattern<DotOp> {
  DotToLinalgGeneric(
      ::mlir::MLIRContext *context,
      std::function<FHEMulOp(mlir::OpBuilder &, mlir::Location, mlir::Type,
                             mlir::Value, mlir::Value)>
          createMulOp,
      std::function<void(DotOp &, FHE::AddEintOp &, FHEMulOp &)>
          forwardOptimizerID)
      : ::mlir::OpRewritePattern<DotOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT),
        createMulOp(createMulOp), forwardOptimizerID(forwardOptimizerID) {}

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
  matchAndRewrite(DotOp dotOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto zeroTensorOp = rewriter.create<mlir::concretelang::FHE::ZeroTensorOp>(
        dotOp.getLoc(), mlir::RankedTensorType::get({1}, dotOp.getType()));

    // Create `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{zeroTensorOp.getType()};
    llvm::SmallVector<mlir::Value, 2> ins{dotOp.getLhs(), dotOp.getRhs()};
    llvm::SmallVector<mlir::Value, 1> outs{zeroTensorOp};
    llvm::SmallVector<mlir::AffineMap, 3> maps{
        mlir::AffineMap::getMultiDimIdentityMap(1, this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(1, this->getContext()),
        mlir::AffineMap::get(1, 0, {rewriter.getAffineConstantExpr(0)},
                             this->getContext())};

    llvm::SmallVector<mlir::utils::IteratorType, 1> itTypes{
        mlir::utils::IteratorType::reduction};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    auto regBuilder = [&](mlir::OpBuilder &nestedBuilder,
                          mlir::Location nestedLoc,
                          mlir::ValueRange blockArgs) {
      auto mul = this->createMulOp(nestedBuilder, dotOp.getLoc(),
                                   dotOp.getResult().getType(), blockArgs[0],
                                   blockArgs[1]);
      mlir::concretelang::FHE::AddEintOp add =
          nestedBuilder.create<mlir::concretelang::FHE::AddEintOp>(
              dotOp.getLoc(), mul, blockArgs[2]);
      nestedBuilder.create<mlir::linalg::YieldOp>(dotOp.getLoc(),
                                                  add.getResult());
      forwardOptimizerID(dotOp, add, mul);
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
    auto res = rewriter.create<mlir::tensor::ExtractOp>(
        dotOp.getLoc(), gop.getResult(0), indexes);

    rewriter.replaceOp(dotOp, {res});

    return ::mlir::success();
  };

private:
  std::function<FHEMulOp(mlir::OpBuilder &, mlir::Location, mlir::Type,
                         mlir::Value, mlir::Value)>
      createMulOp;
  std::function<void(DotOp &, FHE::AddEintOp &, FHEMulOp &)> forwardOptimizerID;
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
    mlir::RankedTensorType lhsTy = ((mlir::Type)linalgOp.getLhs().getType())
                                       .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType rhsTy = ((mlir::Type)linalgOp.getRhs().getType())
                                       .cast<mlir::RankedTensorType>();
    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<FHE::ZeroTensorOp>(
        linalgOp.getLoc(), resultTy, mlir::ValueRange{});

    // Create the affine #maps_0
    llvm::SmallVector<mlir::AffineMap, 3> maps{
        getBroadcastedAffineMap(resultTy, lhsTy, rewriter),
        getBroadcastedAffineMap(resultTy, rhsTy, rewriter),
        getBroadcastedAffineMap(resultTy, resultTy, rewriter),
    };

    // Create the iterator_types
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        resultTy.getShape().size(), mlir::utils::IteratorType::parallel);

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      FHEOp fheOp = nestedBuilder.create<FHEOp>(linalgOp.getLoc(),
                                                resultTy.getElementType(),
                                                blockArgs[0], blockArgs[1]);
      forwardOptimizerID(linalgOp, fheOp);

      nestedBuilder.create<mlir::linalg::YieldOp>(linalgOp.getLoc(),
                                                  fheOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 2> ins{linalgOp.getLhs(), linalgOp.getRhs()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(linalgOp.getLoc(), resTypes,
                                                 ins, outs, maps, iteratorTypes,
                                                 doc, call, bodyBuilder);

    if (linalgOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", linalgOp->getAttr("tile-sizes"));

    rewriter.replaceOp(linalgOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

template <class T> inline mlir::RankedTensorType getRankedTensorType(T v) {
  return ((mlir::Type)v.getType()).cast<mlir::RankedTensorType>();
}

llvm::SmallVector<mlir::utils::IteratorType> parallelIteratorType(int n) {
  return llvm::SmallVector<mlir::utils::IteratorType>(
      n, mlir::utils::IteratorType::parallel);
}

/// This class rewrite pattern transforms any instance of
/// operators `FHELinalg.ApplyMappedLookupTableEintOp` that implements the
/// broadasting rules to an instance of `linalg.generic` with an appropriate
/// region using `FHE.ApplyLookupTableEintOp` operation, an appropriate
/// specification for the iteration dimensions and appropriate operations
/// managing the accumulator of `linalg.generic`.
///
/// Example:
/// %res = "FHELinalg.apply_mapped_lookup_table"(%t, %luts, %map)
/// : (tensor<2x3x!FHE.eint<2>>, tensor<5x4xi64>, tensor<2x3xindex>)
/// -> tensor<2x3x!FHE.eint<2>>
///
/// becomes:
///
/// #map = affine_map<(d0, d1) -> (d0, d1)>
/// %init = linalg.init_tensor [2, 3] : tensor<2x3x!TFHE.glwe<sk?>>
/// %output = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types
/// = ["parallel", "parallel"]} ins(%arg0, %arg2 :
/// tensor<2x3x!TFHE.glwe<sk?>>, tensor<2x3xindex>) outs(%0 :
/// tensor<2x3x!TFHE.glwe<sk?>>) {
///          ^bb0(%arg3: !TFHE.glwe<sk?>, %lut_idx: index, %arg5:
///          !TFHE.glwe<sk?>):  // no predecessors
///          %lut = tensor.extract_slice %arg1[%[[LUTIDX]], 0] [1,4] [1, 1]
///                 : tensor<5x4xi64> to tensor<4xi64>
///          %res  = "TFHE.apply_lookup_table"(%arg3, %[[LUT]])
///                    {baseLogBS = -1 : i32, baseLogKS = -1 : i32,
///                    glweDimension = -1 : i32,
///                      levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS =
///                      -1 : i32, polynomialSize = -1 : i32}
///                 : (!TFHE.glwe<sk?>, tensor<4xi64>) ->
///          !TFHE.glwe<sk?> linalg.yield %res :
///          !TFHE.glwe<sk?>
/// } -> tensor<2x3x!TFHE.glwe<sk?>>

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

    auto input = mappedLookup.getT();
    auto luts = mappedLookup.getLuts();
    auto map = mappedLookup.getMap();

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
    // %arg1 is the lut index (i64)
    // %arg2 is the output element
    auto lambdaBlock = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      auto tElmt = blockArgs[0];
      auto lutIdx = blockArgs[1];

      // %lut = extract_slice %luts[%lutIdx, 0][1, lutSize][1, 1]  :
      // tensor<NxKxi64> to tensor<Kxi64>
      sliceArg offsets{lutIdx, _0_};
      sliceArg sizes{_1_, lutSizeValue};
      sliceArg strides{_1_, _1_};
      auto lutTy = mlir::RankedTensorType::get({static_cast<int64_t>(lutSize)},
                                               lutElmtTy);
      mlir::Value lut = nestedBuilder.create<tensor::ExtractSliceOp>(
          loc, lutTy, luts, offsets, sizes, strides);
      // %res1 = apply_lookup_table %arg0 %lut
      auto lookup = nestedBuilder.create<FHE::ApplyLookupTableEintOp>(
          loc, elementTy, tElmt, lut);
      forwardOptimizerID(mappedLookup, lookup);
      // linalg.yield %res1 : !FHE.eint<2>
      nestedBuilder.create<linalg::YieldOp>(loc, lookup.getResult());
    };

    auto output =
        rewriter.create<FHE::ZeroTensorOp>(loc, resultTy, mlir::ValueRange{});

    // Create the `linalg.g eneric` op
    Types resTys{resultTy};
    Values ins{input, map};
    Values outs{output};
    auto indexOfInput = getBroadcastedAffineMap(resultTy, tensorTy, rewriter);
    AffineMaps affineMaps{indexOfInput, indexOfInput, indexOfInput};
    auto iteratorTypes = parallelIteratorType(resultShape.size());
    auto genericOp = rewriter.create<linalg::GenericOp>(
        loc, resTys, ins, outs, affineMaps, iteratorTypes, lambdaBlock);

    if (mappedLookup->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", mappedLookup->getAttr("tile-sizes"));

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
/// ]
/// #attributes_0 {
///     indexing_maps = #maps_0,
///     iterator_types = ["parallel", "parallel"],
/// }
/// %init = linalg.init_tensor [4, 3]
///            : tensor<4x3x!FHE.eint<2>>
/// %res = linalg.generic {
///     ins(%t, %luts: tensor<4x3x!FHE.eint<p>>)
///     outs(%init : tensor<4x3x!FHE.eint<2>>)
///     {
///         ^bb0(%arg0: !FHE.eint<2>):
///             %i_lut  = linalg.index 0 ; index
///             %lut = tensor.extract_slice  %arg21[%i_lut, 0] [1, lut_size] [1,
///             1] : ... tensor<4xi64> %0 = "TFHE.apply_lookup_table"(%arg0,
///             %lut) {baseLogBS = -1 : i32, baseLogKS = -1 : i32, glweDimension
///             = -1 : i32, levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS
///             = -1 : i32, polynomialSize = -1 : i32} :
///             (!TFHE.glwe<sk?>, tensor<4xi64>) ->
///             !TFHE.glwe<sk?>
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
    mlir::RankedTensorType tensorTy =
        ((mlir::Type)fheLinalgLutOp.getT().getType())
            .cast<mlir::RankedTensorType>();
    auto luts = fheLinalgLutOp.getLuts();
    mlir::RankedTensorType lutsTy = getRankedTensorType(luts);
    auto lutElmtTy = lutsTy.getElementType();
    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<FHE::ZeroTensorOp>(
        fheLinalgLutOp.getLoc(), resultTy, mlir::ValueRange{});

    auto lutsShape = lutsTy.getShape();
    auto lut_size = lutsShape[lutsShape.size() - 1];
    auto indexOfInput = getBroadcastedAffineMap(resultTy, tensorTy, rewriter);
    // Create the affine maps
    llvm::SmallVector<mlir::AffineMap> maps{indexOfInput, indexOfInput};

    // Create the iterator_types
    auto iteratorTypes = parallelIteratorType(resultTy.getShape().size());

    auto integer = [&](auto v) -> mlir::Attribute {
      return rewriter.getI64IntegerAttr(v);
    };

    // We need to know with linalg.generic index to use for lut
    // In broadcast case the lut index is inner dimensions of the tensor index
    auto tensorShape = tensorTy.getShape();
    auto tensorRank = tensorTy.getShape().size();
    auto lutsRank = lutsShape.size() - 1; // do not count inner dim of luts
    auto lutIndexDimAt = tensorRank - lutsRank;
    llvm::SmallVector<uint> indexLutsToLinalg(lutsRank);
    for (auto lutsIndex = 0u; lutsIndex < lutsRank; lutsIndex++) {
      auto tensorIndex = lutIndexDimAt + lutsIndex;
      if (tensorShape[tensorIndex] != lutsShape[lutsIndex]) {
        llvm::errs() << "ERROR: Broadcast only works by having more outer "
                        "dims.\nConflict: "
                     << tensorShape[tensorIndex] << " (tensor dim "
                     << tensorIndex << ") is not compatible with "
                     << lutsShape[lutsIndex] << " (luts dim " << lutsIndex
                     << ")\n\n";
        return ::mlir::LogicalResult::failure();
      };
      indexLutsToLinalg[lutsIndex] = tensorIndex;
    }

    auto _0_ = integer(0);
    auto _1_ = integer(1);
    auto lutSizeValue = integer(lut_size);
    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      auto loc = fheLinalgLutOp.getLoc();
      auto tElmt = blockArgs[0];

      // %lut = extract_slice %luts[%lutIdx, 0][1, lutSize][1, 1]  :
      // tensor<NxKxi64> to tensor<Kxi64>
      auto sliceArgDim = lutsShape.size();
      using sliceArg = llvm::SmallVector<mlir::OpFoldResult>;
      sliceArg offsets(sliceArgDim, _0_);
      auto lutsIndex = 0;
      for (auto index : indexLutsToLinalg) {
        auto offset = nestedBuilder.create<linalg::IndexOp>(loc, index);
        offsets[lutsIndex++] = (mlir::OpFoldResult)offset;
      }
      sliceArg sizes(sliceArgDim, _1_);
      sizes[sliceArgDim - 1] = lutSizeValue;
      sliceArg strides(sliceArgDim, _1_);
      auto lutTy = mlir::RankedTensorType::get({static_cast<int64_t>(lut_size)},
                                               lutElmtTy);
      mlir::Value lut = nestedBuilder.create<tensor::ExtractSliceOp>(
          loc, lutTy, luts, offsets, sizes, strides);
      auto lutOp = nestedBuilder.create<FHE::ApplyLookupTableEintOp>(
          loc, resultTy.getElementType(), tElmt, lut);
      forwardOptimizerID(fheLinalgLutOp, lutOp);

      nestedBuilder.create<mlir::linalg::YieldOp>(loc, lutOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value> ins{fheLinalgLutOp.getT()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(
            fheLinalgLutOp.getLoc(), resTypes, ins, outs, maps, iteratorTypes,
            doc, call, bodyBuilder);

    if (fheLinalgLutOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", fheLinalgLutOp->getAttr("tile-sizes"));

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
        ((mlir::Type)lutOp.getT().getType()).cast<mlir::RankedTensorType>();

    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<FHE::ZeroTensorOp>(
        lutOp.getLoc(), resultTy, mlir::ValueRange{});

    // Create the affine #maps_0
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getMultiDimIdentityMap(tTy.getShape().size(),
                                                this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(resultTy.getShape().size(),
                                                this->getContext()),
    };

    // Create the iterator_types
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        resultTy.getShape().size(), mlir::utils::IteratorType::parallel);

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      mlir::concretelang::FHE::ApplyLookupTableEintOp fheOp =
          nestedBuilder.create<mlir::concretelang::FHE::ApplyLookupTableEintOp>(
              lutOp.getLoc(), resultTy.getElementType(), blockArgs[0],
              lutOp.getLut());
      forwardOptimizerID(lutOp, fheOp);

      nestedBuilder.create<mlir::linalg::YieldOp>(lutOp.getLoc(),
                                                  fheOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 1> ins{lutOp.getT()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(lutOp.getLoc(), resTypes, ins,
                                                 outs, maps, iteratorTypes, doc,
                                                 call, bodyBuilder);

    if (lutOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", lutOp->getAttr("tile-sizes"));

    rewriter.replaceOp(lutOp, {genericOp.getResult(0)});

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
template <typename FHELinalgMatmulOp, typename FHEMulOp>
struct FHELinalgMatmulToLinalgGeneric
    : public mlir::OpRewritePattern<FHELinalgMatmulOp> {
  FHELinalgMatmulToLinalgGeneric(
      mlir::MLIRContext *context,
      std::function<FHEMulOp(mlir::OpBuilder &, mlir::Location, mlir::Type,
                             mlir::Value, mlir::Value)>
          createMulOp,
      std::function<void(FHELinalgMatmulOp &, FHE::AddEintOp &, FHEMulOp &)>
          forwardOptimizerID,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : mlir::OpRewritePattern<FHELinalgMatmulOp>(context, benefit),
        createMulOp(createMulOp), forwardOptimizerID(forwardOptimizerID) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalgMatmulOp matmulOp,
                  mlir::PatternRewriter &rewriter) const override {

    mlir::Location location = matmulOp.getLoc();

    mlir::Value lhs = matmulOp.getLhs();
    mlir::Value rhs = matmulOp.getRhs();
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

    auto iteratorTypes = llvm::SmallVector<mlir::utils::IteratorType, 3>{};

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
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
      }
      iteratorTypes.push_back(mlir::utils::IteratorType::reduction);

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
          iteratorTypes.push_back(mlir::utils::IteratorType::reduction);
        } else {
          iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
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
        iteratorTypes.push_back(mlir::utils::IteratorType::parallel);
      }
      iteratorTypes.push_back(mlir::utils::IteratorType::reduction);

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
      auto multiplication = createMulOp(nestedBuilder, location, outElementType,
                                        blockArgs[0], blockArgs[1]);

      auto addition = nestedBuilder.create<FHE::AddEintOp>(
          location, outElementType, blockArgs[2], multiplication);
      forwardOptimizerID(matmulOp, addition, multiplication);

      nestedBuilder.create<linalg::YieldOp>(location, addition.getResult());
    };

    auto resultTypes = llvm::SmallVector<mlir::Type, 1>{outType};
    mlir::linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
        location, resultTypes, ins, outs, maps, iteratorTypes, regionBuilder);

    if (matmulOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", matmulOp->getAttr("tile-sizes"));

    rewriter.replaceOp(matmulOp, genericOp.getResults());
    return mlir::success();
  };

private:
  std::function<FHEMulOp(mlir::OpBuilder &, mlir::Location, mlir::Type,
                         mlir::Value, mlir::Value)>
      createMulOp;
  std::function<void(FHELinalgMatmulOp &, FHE::AddEintOp &, FHEMulOp &)>
      forwardOptimizerID;
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
        mlir::Operation *newOp;
        if (outputIsTensor) {
          newOp = rewriter.create<FHE::ZeroTensorOp>(location, outputType);
        } else {
          newOp = rewriter.create<FHE::ZeroEintOp>(location, outputType);
        }
        forwardOptimizerID(sumOp, newOp);
        rewriter.replaceOp(sumOp, newOp->getResults());
        return mlir::success();
      }
    }

    auto axesToDestroy = std::unordered_set<int64_t>{};
    for (mlir::Attribute axisAttribute : sumOp.getAxes()) {
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
        } else if (sumOp.getKeepDims()) {
          outputAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
        }
      }
    } else {
      outputAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
    }

    mlir::AffineMap outputMap = mlir::AffineMap::get(
        inputDimensions, 0, outputAffineExpressions, rewriter.getContext());

    auto maps = llvm::SmallVector<mlir::AffineMap, 2>{inputMap, outputMap};

    auto iteratorTypes = llvm::SmallVector<mlir::utils::IteratorType, 3>(
        inputDimensions, mlir::utils::IteratorType::parallel);

    for (int64_t axis : axesToDestroy) {
      iteratorTypes[axis] = mlir::utils::IteratorType::reduction;
    }

    auto regionBuilder = [&](mlir::OpBuilder &nestedBuilder,
                             mlir::Location nestedLoc,
                             mlir::ValueRange blockArgs) {
      mlir::Value lhs = blockArgs[0];
      mlir::Value rhs = blockArgs[1];
      auto addition = nestedBuilder.create<FHE::AddEintOp>(location, lhs, rhs);
      forwardOptimizerID(sumOp, addition);

      nestedBuilder.create<linalg::YieldOp>(location, addition.getResult());
    };

    auto resultTypes = llvm::SmallVector<mlir::Type, 1>{accumulatorType};
    linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
        location, resultTypes, ins, outs, maps, iteratorTypes, regionBuilder);

    if (sumOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", sumOp->getAttr("tile-sizes"));

    mlir::Value accumulation = genericOp.getResult(0);

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

    auto n_dim = inputType.getShape().size();

    mlir::Location location = transposeOp.getLoc();
    // Initialize empty tensor to fill with transpose result
    mlir::Value zeroTensor =
        rewriter.create<FHE::ZeroTensorOp>(location, outputType).getResult();

    std::vector<unsigned int> perms = {};

    mlir::ArrayAttr axes = transposeOp.getAxes();
    if (axes.empty()) {
      for (int i = n_dim - 1; i >= 0; i--) {
        perms.push_back(i);
      }
    } else {
      for (mlir::Attribute axisAttribute : axes) {
        int64_t axis = axisAttribute.cast<mlir::IntegerAttr>().getInt();
        perms.push_back(axis);
      }
    }

    llvm::SmallVector<mlir::Type, 1> resultTypes{zeroTensor.getType()};
    auto ins = llvm::SmallVector<mlir::Value, 1>{input};
    auto outs = llvm::SmallVector<mlir::Value, 1>{zeroTensor};
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getMultiDimIdentityMap(n_dim, this->getContext()),
        mlir::AffineMap::getPermutationMap(perms, this->getContext()),
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

    linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
        location, resultTypes, ins, outs, maps, iteratorTypes, regionBuilder);

    if (transposeOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", transposeOp->getAttr("tile-sizes"));

    rewriter.replaceOp(transposeOp, genericOp.getResults());
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
    size_t axis = op.getAxis();

    mlir::Value output = op.getResult();
    auto outputType = output.getType().dyn_cast<mlir::TensorType>();

    llvm::ArrayRef<int64_t> outputShape = outputType.getShape();
    size_t outputDimensions = outputShape.size();

    mlir::Value result =
        rewriter
            .create<tensor::EmptyOp>(location, outputShape,
                                     outputType.getElementType())
            .getResult();

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
      mlir::DenseI64ArrayAttr offsetsAttr =
          rewriter.getDenseI64ArrayAttr(offsets);
      mlir::DenseI64ArrayAttr sizesAttr = rewriter.getDenseI64ArrayAttr(sizes);
      mlir::DenseI64ArrayAttr stridesAttr =
          rewriter.getDenseI64ArrayAttr(strides);

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

  mlir::Value paddedInput = b.create<mlir::tensor::PadOp>(
      loc, rankedTensorType, input, lowPaddings, highPaddings, pad);

  return paddedInput;
}

mlir::Value extractContiguous4DSlice(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc, mlir::Value input,
                                     mlir::RankedTensorType resultType,
                                     llvm::SmallVector<int64_t, 4> sizes,
                                     llvm::SmallVector<int64_t, 4> offsets) {
  return rewriter
      .create<mlir::tensor::ExtractSliceOp>(
          loc, resultType, input,
          // offset
          llvm::SmallVector<mlir::OpFoldResult, 4>{
              rewriter.getI64IntegerAttr(offsets[0]),
              rewriter.getI64IntegerAttr(offsets[1]),
              rewriter.getI64IntegerAttr(offsets[2]),
              rewriter.getI64IntegerAttr(offsets[3]),
          },
          // sizes
          llvm::SmallVector<mlir::OpFoldResult, 4>{
              rewriter.getI64IntegerAttr(sizes[0]),
              rewriter.getI64IntegerAttr(sizes[1]),
              rewriter.getI64IntegerAttr(sizes[2]),
              rewriter.getI64IntegerAttr(sizes[3]),
          },
          // strides
          llvm::SmallVector<mlir::OpFoldResult, 4>{
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(1),
              rewriter.getI64IntegerAttr(1),
          })
      .getResult();
}

/// Create operations for grouped convolution. This will slice the input,
/// weight, and output tensors to apply separate conv2d operations.
mlir::LogicalResult createGroupedConv2D(
    mlir::PatternRewriter &rewriter,
    mlir::concretelang::FHELinalg::Conv2dOp &conv2dOp, mlir::Value paddedInput,
    mlir::Value weight, mlir::Value outputTensor,
    mlir::DenseIntElementsAttr stridesAttr,
    mlir::DenseIntElementsAttr dilationsAttr,
    llvm::ArrayRef<mlir::NamedAttribute> namedAttr, int64_t group) {

  mlir::RankedTensorType inputTy =
      paddedInput.getType().cast<mlir::RankedTensorType>();
  mlir::Type inputElemTy = inputTy.getElementType();
  llvm::ArrayRef<int64_t> inputShape = inputTy.getShape();
  llvm::SmallVector<int64_t, 4> inputSliceSizes(
      {inputShape[0], inputShape[1] / group, inputShape[2], inputShape[3]});

  mlir::RankedTensorType weightTy =
      weight.getType().cast<mlir::RankedTensorType>();
  mlir::Type weightElemTy = weightTy.getElementType();
  llvm::ArrayRef<int64_t> weightShape = weightTy.getShape();
  llvm::SmallVector<int64_t, 4> weightSliceSizes(
      {weightShape[0] / group, weightShape[1], weightShape[2], weightShape[3]});

  mlir::RankedTensorType resultTy =
      conv2dOp.getResult().getType().cast<mlir::RankedTensorType>();
  llvm::ArrayRef<int64_t> resultShape = resultTy.getShape();
  llvm::SmallVector<int64_t, 4> sliceResultSizes = {
      resultShape[0], weightSliceSizes[0], resultShape[2], resultShape[3]};
  mlir::RankedTensorType sliceResultType =
      mlir::RankedTensorType::get(sliceResultSizes, inputElemTy);

  // slice the input, weight, and output to apply different convolutions and
  // store their outputs in a single result found in `finalResult`
  mlir::Value finalResult = outputTensor;
  for (int g = 0; g < group; g++) {
    // input[:][g * (input_C / group) : (g + 1) * (input_C / group)][:][:]
    mlir::Value inputSlice = extractContiguous4DSlice(
        rewriter, conv2dOp.getLoc(), paddedInput,
        mlir::RankedTensorType::get(inputSliceSizes, inputElemTy),
        inputSliceSizes, {0, g * inputSliceSizes[1], 0, 0});
    // weight[g * (weight_F / group) : (g + 1) * (weight_F / group)][:][:][:]
    mlir::Value weightSlice = extractContiguous4DSlice(
        rewriter, conv2dOp.getLoc(), weight,
        mlir::RankedTensorType::get(weightSliceSizes, weightElemTy),
        weightSliceSizes, {g * weightSliceSizes[0], 0, 0, 0});
    // bias[:][g * (weight_F / group) : (g + 1) * (weight_F / group)][:][:]
    mlir::Value biasSlice = extractContiguous4DSlice(
        rewriter, conv2dOp.getLoc(), outputTensor, sliceResultType,
        sliceResultSizes, {0, g * sliceResultSizes[1], 0, 0});
    // slices are currently causing issues during scf bufferization, so we are
    // trying to avoid slices here by creating a new tensor and adding the bias
    // to it
    mlir::RankedTensorType biasSliceType =
        biasSlice.getType().cast<mlir::RankedTensorType>();
    auto biasUnslicedInit =
        rewriter
            .create<mlir::concretelang::FHE::ZeroTensorOp>(
                conv2dOp.getLoc(),
                mlir::RankedTensorType::get(biasSliceType.getShape(),
                                            biasSliceType.getElementType()))
            .getResult();
    auto biasUnsliced =
        rewriter.create<mlir::concretelang::FHELinalg::AddEintOp>(
            conv2dOp.getLoc(), biasUnslicedInit, biasSlice);
    forwardOptimizerID(conv2dOp, biasUnsliced);

    // apply conv
    mlir::Value convResult =
        rewriter
            .create<mlir::linalg::Conv2DNchwFchwOp>(
                conv2dOp.getLoc(), sliceResultType,
                mlir::ValueRange{inputSlice, weightSlice},
                biasUnsliced.getResult(), stridesAttr, dilationsAttr, namedAttr)
            .getResult(0);
    // insert result of a single conv in the final result
    finalResult =
        rewriter
            .create<mlir::tensor::InsertSliceOp>(
                conv2dOp.getLoc(), convResult, finalResult,
                llvm::SmallVector<mlir::OpFoldResult, 4>{
                    rewriter.getI64IntegerAttr(0),
                    rewriter.getI64IntegerAttr(g * sliceResultSizes[1]),
                    rewriter.getI64IntegerAttr(0),
                    rewriter.getI64IntegerAttr(0),
                },
                llvm::SmallVector<mlir::OpFoldResult, 4>{
                    rewriter.getI64IntegerAttr(sliceResultSizes[0]),
                    rewriter.getI64IntegerAttr(sliceResultSizes[1]),
                    rewriter.getI64IntegerAttr(sliceResultSizes[2]),
                    rewriter.getI64IntegerAttr(sliceResultSizes[3]),
                },
                llvm::SmallVector<mlir::OpFoldResult, 4>{
                    rewriter.getI64IntegerAttr(1),
                    rewriter.getI64IntegerAttr(1),
                    rewriter.getI64IntegerAttr(1),
                    rewriter.getI64IntegerAttr(1),
                })
            .getResult();
  }

  rewriter.replaceOp(conv2dOp, finalResult);
  return mlir::success();
}

bool isZeroConstant(mlir::Value value) {
  auto cst =
      mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(value.getDefiningOp());
  if (cst == nullptr)
    return false;
  auto values = cst->getAttrOfType<mlir::DenseIntElementsAttr>("value");
  if (values == nullptr)
    return false;
  for (auto v : values) {
    if (v != 0)
      return false;
  }
  return true;
}

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.conv2d` to one or multiple instances of
/// `linalg.conv_2d_nchw_fchw`. The transformation consists of padding the input
/// tensor, and initializing the output tensor with bias values if any. Multiple
/// linalng conv operations can be generated, and their output concatenated in
/// the case of grouped convolution
struct FHELinalgConv2dToLinalgConv2d
    : public ::mlir::OpRewritePattern<mlir::concretelang::FHELinalg::Conv2dOp> {
  FHELinalgConv2dToLinalgConv2d(::mlir::MLIRContext *context)
      : ::mlir::OpRewritePattern<::mlir::concretelang::FHELinalg::Conv2dOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(::mlir::concretelang::FHELinalg::Conv2dOp conv2dOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    auto optimizerIdAttr =
        conv2dOp->getAttrOfType<mlir::IntegerAttr>("TFHE.OId");

    // Create named attr for custom linalg op
    std::vector<mlir::NamedAttribute> addOpDict = {rewriter.getNamedAttr(
        "op", rewriter.getStringAttr(
                  mlir::concretelang::FHE::AddEintOp::getOperationName()))};
    std::vector<mlir::NamedAttribute> mulOpDict = {rewriter.getNamedAttr(
        "op", rewriter.getStringAttr(
                  mlir::concretelang::FHE::MulEintIntOp::getOperationName()))};
    std::vector<mlir::NamedAttribute> opAttrs;
    if (optimizerIdAttr != nullptr) {
      opAttrs.push_back(rewriter.getNamedAttr("TFHE.OId", optimizerIdAttr));
    }
    auto opNamedAttrs =
        rewriter.getNamedAttr("op_attrs", rewriter.getDictionaryAttr(opAttrs));

    addOpDict.push_back(opNamedAttrs);
    mulOpDict.push_back(opNamedAttrs);

    auto addOpAttr =
        rewriter.getNamedAttr("add", rewriter.getDictionaryAttr(addOpDict));
    auto mulOpAttr =
        rewriter.getNamedAttr("mul", rewriter.getDictionaryAttr(mulOpDict));
    std::vector<mlir::NamedAttribute> namedAttr({addOpAttr, mulOpAttr});

    mlir::Location loc = conv2dOp->getLoc();
    mlir::Value input =
        conv2dOp.getInput(); /* shape: Batch*Channels*Height*Width */
    mlir::Value weight =
        conv2dOp.getWeight(); /* shape: Filters*Channels*Height*Width */

    mlir::Type inputElementType =
        input.getType().cast<mlir::RankedTensorType>().getElementType();

    // Attriutes are assumed to be correct after passing the verification
    mlir::SmallVector<int64_t, 4> paddingInts =
        mlir::concretelang::FHELinalg::getPaddingFromConv2d(conv2dOp);
    mlir::SmallVector<int64_t, 2> stridesInts =
        mlir::concretelang::FHELinalg::getStridesFromConv2d(conv2dOp);
    mlir::SmallVector<int64_t, 2> dilationsInts =
        mlir::concretelang::FHELinalg::getDilationsFromConv2d(conv2dOp);
    int64_t group = mlir::concretelang::FHELinalg::getGroupFromConv2d(conv2dOp);

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
    forwardOptimizerID(conv2dOp, initTensor.getDefiningOp());
    // Since linalg doesn't support a bias in the conv operation, we
    // initialize the output tensor to the bias values, so that conv results
    // get accumulated to it
    mlir::Value bias = conv2dOp.getBias(); /* optional of shape: Filters */
    mlir::Value biasInitTensor = initTensor;
    if (bias && !isZeroConstant(bias)) {
      // Fill the output tensor with bias values
      auto resultRank =
          initTensor.getType().cast<mlir::RankedTensorType>().getRank();
      mlir::SmallVector<mlir::AffineMap> indexingMaps = {
          mlir::AffineMap::get(resultRank, 0, rewriter.getAffineDimExpr(1),
                               rewriter.getContext()),
          rewriter.getMultiDimIdentityMap(resultRank)};
      mlir::SmallVector<mlir::utils::IteratorType> iteratorTypes(
          resultRank, mlir::utils::IteratorType::parallel);
      biasInitTensor =
          rewriter
              .create<mlir::linalg::GenericOp>(
                  loc, initTensor.getType(), bias, initTensor, indexingMaps,
                  iteratorTypes,
                  [&](mlir::OpBuilder &b, mlir::Location loc,
                      mlir::ValueRange args) {
                    auto encryptedBias =
                        b.create<mlir::concretelang::FHE::AddEintIntOp>(
                            loc, args[1], args[0]);
                    forwardOptimizerID(conv2dOp, encryptedBias);
                    b.create<mlir::linalg::YieldOp>(loc,
                                                    encryptedBias.getResult());
                  })
              .getResult(0);
    }

    auto stridesAttr = rewriter.getI64VectorAttr(stridesInts);
    auto dilationsAttr = rewriter.getI64VectorAttr(dilationsInts);

    // we can directly use linalg::Conv2DNchwFchwOp if group is equal to 1,
    // but since there is no support for groups in linalg conv operations, we
    // need to slice the different tensors and apply multiple convolution in
    // case group is greater than 1
    if (group == 1) {
      rewriter.replaceOpWithNewOp<mlir::linalg::Conv2DNchwFchwOp>(
          conv2dOp, biasInitTensor.getType(),
          mlir::ValueRange{paddedInput, weight}, biasInitTensor, stridesAttr,
          dilationsAttr, namedAttr);
      return mlir::success();
    }
    return createGroupedConv2D(rewriter, conv2dOp, paddedInput, weight,
                               biasInitTensor, stridesAttr, dilationsAttr,
                               namedAttr, group);
  };
};

/// This rewrite pattern transforms all instances
/// of `FHELinalg.maxpool2d` to `linalg.pooling_ncw_max`.
struct FHELinalgMaxpool2dToLinalgMaxpool2d
    : public mlir::OpRewritePattern<FHELinalg::Maxpool2dOp> {

  FHELinalgMaxpool2dToLinalgMaxpool2d(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHELinalg::Maxpool2dOp>(context) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalg::Maxpool2dOp maxpool2dOp,
                  mlir::PatternRewriter &rewriter) const override {
    auto optimizerIdAttr =
        maxpool2dOp->getAttrOfType<mlir::DenseI32ArrayAttr>("TFHE.OId");

    const mlir::Location loc = maxpool2dOp->getLoc();

    std::vector<mlir::NamedAttribute> maxOpDict = {rewriter.getNamedAttr(
        "op", rewriter.getStringAttr(
                  mlir::concretelang::FHE::MaxEintOp::getOperationName()))};
    std::vector<mlir::NamedAttribute> opAttrs;
    if (optimizerIdAttr != nullptr) {
      opAttrs.push_back(rewriter.getNamedAttr("TFHE.OId", optimizerIdAttr));
    }
    auto opNamedAttrs =
        rewriter.getNamedAttr("op_attrs", rewriter.getDictionaryAttr(opAttrs));
    maxOpDict.push_back(opNamedAttrs);
    const mlir::NamedAttribute maxOpAttr = rewriter.getNamedAttr(
        "max_signed", rewriter.getDictionaryAttr(maxOpDict));

    const auto outputTy =
        maxpool2dOp->getResult(0).getType().cast<mlir::RankedTensorType>();
    const auto outputElementTy =
        outputTy.getElementType().cast<FHE::FheIntegerInterface>();

    auto output = rewriter.create<FHE::ZeroTensorOp>(loc, outputTy).getResult();

    if (outputElementTy.isSigned()) {
      const int64_t outputBitWidth = outputElementTy.getWidth();
      const int64_t offsetValue = 1 << (outputBitWidth - 2);

      const mlir::Type offsetType =
          mlir::IntegerType::get(this->getContext(), outputBitWidth + 1);
      const mlir::Type offsetTensorType =
          mlir::RankedTensorType::get({1}, offsetType);

      const llvm::SmallVector<mlir::Attribute> offsetTensorAttr = {
          mlir::IntegerAttr::get(offsetType, offsetValue)};
      const mlir::Attribute offsetAttr =
          mlir::DenseElementsAttr::get(offsetTensorType, offsetTensorAttr);

      const mlir::Value offset =
          rewriter.create<mlir::arith::ConstantOp>(loc, offsetAttr);

      auto subOp =
          rewriter.create<FHELinalg::SubEintIntOp>(loc, output, offset);
      if (optimizerIdAttr != nullptr) {
        assert(optimizerIdAttr.size() == 3);
        output.getDefiningOp()->setAttr(
            "TFHE.OId", rewriter.getI32IntegerAttr(optimizerIdAttr[2]));
        subOp->setAttr("TFHE.OId",
                       rewriter.getI32IntegerAttr(optimizerIdAttr[2]));
      }
      output = subOp.getResult();
    } else {
      if (optimizerIdAttr != nullptr) {
        output.getDefiningOp()->setAttr(
            "TFHE.OId", rewriter.getI32IntegerAttr(optimizerIdAttr[0]));
      }
    }

    const mlir::DenseElementsAttr kernelShapeAttr =
        maxpool2dOp.getKernelShape();
    const auto kernelShape =
        llvm::SmallVector<int64_t, 2>(kernelShapeAttr.value_begin<int64_t>(),
                                      kernelShapeAttr.value_end<int64_t>());

    const mlir::Value kernel =
        rewriter
            .create<mlir::tensor::EmptyOp>(
                loc, kernelShape,
                mlir::IntegerType::get(this->getContext(), 64))
            .getResult();

    const mlir::DenseIntElementsAttr defaultAttr =
        rewriter.getI64VectorAttr({1, 1});

    const mlir::DenseIntElementsAttr stridesAttr =
        maxpool2dOp.getDilations().value_or(defaultAttr);
    const mlir::DenseIntElementsAttr dilationsAttr =
        maxpool2dOp.getDilations().value_or(defaultAttr);

    rewriter.replaceOpWithNewOp<mlir::linalg::PoolingNchwMaxOp>(
        maxpool2dOp, outputTy, mlir::ValueRange{maxpool2dOp.getInput(), kernel},
        output, stridesAttr, dilationsAttr,
        llvm::ArrayRef<mlir::NamedAttribute>({maxOpAttr}));

    return mlir::success();
  };
};

/// This template rewrite pattern transforms any instance of
/// operators `FHELinalg.UNARY_OP` to an instance of `linalg.generic` with an
/// appropriate region using `FHE.UNARY_OP` operation, an appropriate
/// specification for the iteration dimensions and appropriate operations
/// managing the accumulator of `linalg.generic`.
///
/// Example:
///
/// FHELinalg.UNARY_OP(%tensor):
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
///             %0 = FHE.UNARY_OP(%arg0): !FHE.eint<p> -> !FHE.eint<p'>
///         linalg.yield %0 : !FHE.eint<p'>
///     }
/// }
///

template <typename FHELinalgOp, typename FHEOp>
struct FHELinalgUnaryOpToLinalgGeneric
    : public mlir::OpRewritePattern<FHELinalgOp> {
  FHELinalgUnaryOpToLinalgGeneric(
      ::mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : ::mlir::OpRewritePattern<FHELinalgOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(FHELinalgOp linalgOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)linalgOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType tensorTy =
        ((mlir::Type)linalgOp.getInput().getType())
            .cast<mlir::RankedTensorType>();

    auto loc = linalgOp.getLoc();

    //  linalg.init_tensor for initial value
    mlir::Value init =
        rewriter.create<FHE::ZeroTensorOp>(loc, resultTy, mlir::ValueRange{});

    // Create the affine #maps_0
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getMultiDimIdentityMap(tensorTy.getShape().size(),
                                                this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(resultTy.getShape().size(),
                                                this->getContext()),
    };

    // Create the iterator_types
    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        resultTy.getShape().size(), mlir::utils::IteratorType::parallel);

    // Create the body of the `linalg.generic` op
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      FHEOp fheOp = nestedBuilder.create<FHEOp>(
          linalgOp.getLoc(), resultTy.getElementType(), blockArgs[0]);
      forwardOptimizerID(linalgOp, fheOp);

      nestedBuilder.create<mlir::linalg::YieldOp>(linalgOp.getLoc(),
                                                  fheOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 1> ins{linalgOp.getInput()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(linalgOp.getLoc(), resTypes,
                                                 ins, outs, maps, iteratorTypes,
                                                 doc, call, bodyBuilder);

    if (linalgOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", linalgOp->getAttr("tile-sizes"));

    rewriter.replaceOp(linalgOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

// Replaces a `optimizer.partition_frontier` operation with a tensor
// operand and a tensor result with a `linalg.generic` operation
// applying a `optimizer.partition_frontier` operation with scalar
// operands.
struct TensorPartitionFrontierOpToLinalgGeneric
    : public mlir::OpRewritePattern<
          mlir::concretelang::Optimizer::PartitionFrontierOp> {
  TensorPartitionFrontierOpToLinalgGeneric(
      ::mlir::MLIRContext *context,
      mlir::PatternBenefit benefit =
          mlir::concretelang::DEFAULT_PATTERN_BENEFIT)
      : ::mlir::OpRewritePattern<
            mlir::concretelang::Optimizer::PartitionFrontierOp>(context,
                                                                benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Optimizer::PartitionFrontierOp pfOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        pfOp.getResult().getType().cast<mlir::RankedTensorType>();
    mlir::RankedTensorType tensorTy =
        pfOp.getInput().getType().cast<mlir::RankedTensorType>();

    mlir::Value init = rewriter.create<mlir::tensor::EmptyOp>(
        pfOp.getLoc(), resultTy, mlir::ValueRange{});

    // Create affine maps and iterator types for an embarrassingly
    // parallel op
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        mlir::AffineMap::getMultiDimIdentityMap(tensorTy.getShape().size(),
                                                this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(resultTy.getShape().size(),
                                                this->getContext()),
    };

    llvm::SmallVector<mlir::utils::IteratorType> iteratorTypes(
        resultTy.getShape().size(), mlir::utils::IteratorType::parallel);

    // Create the body of the `linalg.generic` op applying a
    // `tensor.partition_frontier` op on the scalar arguments
    auto bodyBuilder = [&](mlir::OpBuilder &nestedBuilder,
                           mlir::Location nestedLoc,
                           mlir::ValueRange blockArgs) {
      mlir::concretelang::Optimizer::PartitionFrontierOp scalarOp =
          nestedBuilder
              .create<mlir::concretelang::Optimizer::PartitionFrontierOp>(
                  pfOp.getLoc(), resultTy.getElementType(), blockArgs[0],
                  pfOp->getAttrs());

      nestedBuilder.create<mlir::linalg::YieldOp>(pfOp.getLoc(),
                                                  scalarOp.getResult());
    };

    // Create the `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{init.getType()};
    llvm::SmallVector<mlir::Value, 1> ins{pfOp.getInput()};
    llvm::SmallVector<mlir::Value, 1> outs{init};
    llvm::StringRef doc{""};
    llvm::StringRef call{""};

    mlir::linalg::GenericOp genericOp =
        rewriter.create<mlir::linalg::GenericOp>(pfOp.getLoc(), resTypes, ins,
                                                 outs, maps, iteratorTypes, doc,
                                                 call, bodyBuilder);

    if (pfOp->hasAttr("tile-sizes"))
      genericOp->setAttr("tile-sizes", pfOp->getAttr("tile-sizes"));

    rewriter.replaceOp(pfOp, {genericOp.getResult(0)});

    return ::mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.fancy_index` to an instance of `tensor.generate`.
///
/// Example:
///
///   %output = "FHELinalg.fancy_index"(%input, %indices) :
///     (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
///
/// becomes:
///
///   %output = tensor.generate  {
///     ^bb0(%i: index):
///       %index = tensor.extract %indices[%i] : tensor<3xindex>
///       %element = tensor.extract %input[%index] : tensor<5x!FHE.eint<6>>
///       tensor.yield %element : !FHE.eint<6>
///     } : tensor<3x!FHE.eint<6>>
///
struct FancyIndexToTensorGenerate
    : public mlir::OpRewritePattern<FHELinalg::FancyIndexOp> {
  FancyIndexToTensorGenerate(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHELinalg::FancyIndexOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalg::FancyIndexOp fancyIndexOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto location = fancyIndexOp.getLoc();

    auto input = fancyIndexOp.getInput();
    auto indices = fancyIndexOp.getIndices();
    auto output = fancyIndexOp.getOutput();

    auto inputType = input.getType().dyn_cast<mlir::RankedTensorType>();
    auto outputType = output.getType().dyn_cast<mlir::RankedTensorType>();

    auto inputShape = inputType.getShape();
    auto inputDimensions = inputShape.size();
    auto inputIsVector = inputDimensions == 1;
    auto inputElementType = inputType.getElementType();

    auto dynamicExtents = mlir::ValueRange();
    auto body = [=](mlir::OpBuilder &builder, mlir::Location location,
                    mlir::ValueRange args) {
      if (inputIsVector) {
        auto index = builder.create<tensor::ExtractOp>(
            location, builder.getIndexType(), indices, args);

        auto result = builder.create<tensor::ExtractOp>(
            location, inputElementType, input, index.getResult());
        result->setAttrs(fancyIndexOp->getAttrs());

        builder.create<tensor::YieldOp>(location, result);
      } else {
        auto index = llvm::SmallVector<mlir::Value>();
        auto baseArgs = std::vector<mlir::Value>(args.begin(), args.end());
        for (size_t i = 0; i < inputShape.size(); i++) {
          baseArgs.push_back(builder.create<arith::ConstantOp>(
              location, builder.getIndexType(), builder.getIndexAttr(i)));
          index.push_back(builder.create<tensor::ExtractOp>(
              location, builder.getIndexType(), indices, baseArgs));
          baseArgs.pop_back();
        }

        auto result = builder.create<tensor::ExtractOp>(
            location, inputElementType, input, index);
        result->setAttrs(fancyIndexOp->getAttrs());

        builder.create<tensor::YieldOp>(location, result);
      }
    };
    auto result = rewriter.create<tensor::GenerateOp>(location, outputType,
                                                      dynamicExtents, body);
    result->setAttrs(fancyIndexOp->getAttrs());

    rewriter.replaceOp(fancyIndexOp, {result});
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.fancy_assign` to an instance of `scf.forall`.
///
/// Example:
///
///   %output = "FHELinalg.fancy_assign"(%input, %indices, %values) :
///     (tensor<5x!FHE.eint<6>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) ->
///     tensor<5x!FHE.eint<6>>
///
/// becomes:
///
///   %0 = scf.forall (%i) in (3) shared_outs(%output = %input)
///       -> (tensor<5x!FHE.eint<6>>) {
///     %index = tensor.extract %indices[%i] : tensor<3xindex>
///     %value = tensor.extract %values[%i] : tensor<3x!FHE.eint<6>>
///     %value_slice = tensor.from_elements %value : tensor<1x!FHE.eint<6>>
///     scf.forall.in_parallel {
///       tensor.parallel_insert_slice
///         %value_slice into %output[%index][1][1]
///         : tensor<1x!FHE.eint<6>> into tensor<5x!FHE.eint<6>>
///     }
///   }
///
struct FancyAssignToSfcForall
    : public mlir::OpRewritePattern<FHELinalg::FancyAssignOp> {
  FancyAssignToSfcForall(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHELinalg::FancyAssignOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalg::FancyAssignOp fancyAssignOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto input = fancyAssignOp.getInput();
    auto indices = fancyAssignOp.getIndices();
    auto values = fancyAssignOp.getValues();

    auto inputType = input.getType().dyn_cast<mlir::RankedTensorType>();
    auto valuesType =
        fancyAssignOp.getValues().getType().cast<mlir::RankedTensorType>();

    auto inputShape = inputType.getShape();
    auto inputDimensions = inputShape.size();
    auto inputIsVector = inputDimensions == 1;
    auto inputElementType = inputType.getElementType();

    auto upperBounds = llvm::SmallVector<mlir::OpFoldResult>();
    for (auto dimension : valuesType.getShape()) {
      upperBounds.push_back(
          mlir::OpFoldResult(rewriter.getIndexAttr(dimension)));
    }

    auto body = [=](mlir::OpBuilder &builder, mlir::Location location,
                    mlir::ValueRange args) {
      auto output = args[args.size() - 1];
      auto loopArgs = args.take_front(args.size() - 1);

      std::vector<mlir::Value> index;
      mlir::Value element;

      if (inputIsVector) {
        index.push_back(builder.create<tensor::ExtractOp>(
            location, builder.getIndexType(), indices, loopArgs));

        element = builder
                      .create<tensor::ExtractOp>(location, inputElementType,
                                                 values, loopArgs)
                      .getResult();
      } else {
        auto baseArgs =
            std::vector<mlir::Value>(loopArgs.begin(), loopArgs.end());

        for (size_t i = 0; i < inputShape.size(); i++) {
          baseArgs.push_back(builder.create<arith::ConstantOp>(
              location, builder.getIndexType(), builder.getIndexAttr(i)));
          index.push_back(builder.create<tensor::ExtractOp>(
              location, builder.getIndexType(), indices, baseArgs));
          baseArgs.pop_back();
        }

        element = builder
                      .create<tensor::ExtractOp>(location, inputElementType,
                                                 values, loopArgs)
                      .getResult();
      }

      if (!element.getType().isa<mlir::TensorType>()) {
        element =
            builder.create<mlir::tensor::FromElementsOp>(location, element)
                .getResult();
      }

      auto offsets = std::vector<mlir::OpFoldResult>();
      auto sizes = std::vector<mlir::OpFoldResult>();
      auto strides = std::vector<mlir::OpFoldResult>();

      for (size_t i = 0; i < index.size(); i++) {
        offsets.push_back(mlir::OpFoldResult(index[i]));
        sizes.push_back(mlir::OpFoldResult(builder.getIndexAttr(1)));
        strides.push_back(mlir::OpFoldResult(builder.getIndexAttr(1)));
      }

      auto inParallelOp = builder.create<mlir::scf::InParallelOp>(location);
      builder.setInsertionPointToStart(inParallelOp.getBody());

      builder.create<mlir::tensor::ParallelInsertSliceOp>(
          location, element, output, offsets, sizes, strides);
    };

    auto forallOp = rewriter.create<mlir::scf::ForallOp>(
        fancyAssignOp.getLoc(), upperBounds,
        mlir::ValueRange{fancyAssignOp.getInput()}, std::nullopt, body);

    rewriter.replaceOp(fancyAssignOp, forallOp->getResult(0));
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `arith.index_cast` and `arith.extsi` to an instance of `linalg.generic`.
///
/// Example:
///
///   %output = arith.index_cast %input : tensor<3xi4> to tensor<3xindex>
///
/// becomes:
///
///   #map = affine_map<(d0) -> (d0)>
///
///   %empty = tensor.empty() : tensor<3xindex>
///   %output = linalg.generic
///     {
///       indexing_maps = [#map, #map],
///       iterator_types = ["parallel"]
///     }
///     ins(%input : tensor<3xi4>)
///     outs(%empty : tensor<3xindex>)
///     {
///
///       ^bb0(%element: i4, %output: index):
///           %casted = arith.index_cast %element : i4 to index
///           linalg.yield %casted : index
///
///     } -> tensor<3xindex>
///
template <typename Op>
struct TensorMapToLinalgGeneric : public mlir::OpRewritePattern<Op> {
  TensorMapToLinalgGeneric(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<Op>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  ::mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {

    mlir::Location location = op.getLoc();

    mlir::Value input = op.getOperand();
    mlir::Value output = op.getResult();

    auto inputType = input.getType().dyn_cast_or_null<mlir::RankedTensorType>();
    auto outputType =
        output.getType().dyn_cast_or_null<mlir::RankedTensorType>();

    if (!inputType || !outputType) {
      return mlir::failure();
    }

    llvm::ArrayRef<int64_t> inputShape = inputType.getShape();
    int64_t inputDimensions = inputShape.size();

    mlir::Value empty =
        rewriter
            .create<tensor::EmptyOp>(location, outputType.getShape(),
                                     outputType.getElementType())
            .getResult();

    auto ins = llvm::SmallVector<mlir::Value, 1>{input};
    auto outs = llvm::SmallVector<mlir::Value, 1>{empty};

    mlir::AffineMap identityMap = mlir::AffineMap::getMultiDimIdentityMap(
        inputDimensions, this->getContext());

    auto maps = llvm::SmallVector<mlir::AffineMap, 2>{identityMap, identityMap};

    auto iteratorTypes = llvm::SmallVector<mlir::utils::IteratorType, 3>(
        inputDimensions, mlir::utils::IteratorType::parallel);

    auto regionBuilder = [&](mlir::OpBuilder &nestedBuilder,
                             mlir::Location nestedLoc,
                             mlir::ValueRange blockArgs) {
      mlir::Value value = blockArgs[0];
      auto cast = nestedBuilder.create<Op>(location,
                                           outputType.getElementType(), value);
      nestedBuilder.create<linalg::YieldOp>(location, cast.getResult());
    };

    auto resultTypes = llvm::SmallVector<mlir::Type, 1>{outputType};
    linalg::GenericOp genericOp = rewriter.create<linalg::GenericOp>(
        location, resultTypes, ins, outs, maps, iteratorTypes, regionBuilder);

    if (op->hasAttr("tile-sizes")) {
      genericOp->setAttr("tile-sizes", op->getAttr("tile-sizes"));
    }

    mlir::Value replacement = genericOp.getResult(0);

    rewriter.replaceOp(op, {replacement});
    return mlir::success();
  };
};

/// This rewrite pattern transforms any instance of operators
/// `FHELinalg.broadcast` to an instance of `linalg.generic`.
///
/// Example:
///
///   %output =  "FHELinalg.broadcast"(%input)
///     : (tensor<3xindex>) -> tensor<4x3xindex>
///
/// becomes:
///
///   #map1 = affine_map<(d0, d1) -> (d1)>
///   #map2 = affine_map<(d0, d1) -> (d0, d1)>
///
///   %empty = tensor.empty() : tensor<4x3xindex>
///   %output = linalg.generic
///     {
///       indexing_maps = [#map1, #map2],
///       iterator_types = ["parallel", "parallel"]
///     }
///     ins(%input : tensor<3xindex>)
///     outs(%empty : tensor<4x3xindex>)
///     {
///
///       ^bb0(%element: index, %output: index):
///           linalg.yield %element : index
///
///     } -> tensor<4x3xindex>
///
struct BroadcastToLinalgGeneric
    : public mlir::OpRewritePattern<FHELinalg::BroadcastOp> {
  BroadcastToLinalgGeneric(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHELinalg::BroadcastOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalg::BroadcastOp broadcastOp,
                  mlir::PatternRewriter &rewriter) const override {

    auto location = broadcastOp.getLoc();

    auto input = broadcastOp.getInput();
    auto output = broadcastOp.getOutput();

    auto inputType = input.getType().dyn_cast<mlir::RankedTensorType>();
    auto outputType = output.getType().dyn_cast<mlir::RankedTensorType>();

    auto inputShape = inputType.getShape();
    auto outputShape = outputType.getShape();

    auto inputDimensions = inputShape.size();
    auto outputDimensions = outputShape.size();

    auto emptyTensorOp = rewriter.create<tensor::EmptyOp>(location, outputType,
                                                          mlir::ValueRange{});

    auto inputAffineExpressions = llvm::SmallVector<mlir::AffineExpr, 3>{};
    for (size_t i = 0; i < inputDimensions; i++) {
      if (inputShape[i] == 1) {
        inputAffineExpressions.push_back(rewriter.getAffineConstantExpr(0));
      } else {
        auto dim = i + (outputDimensions - inputDimensions);
        inputAffineExpressions.push_back(rewriter.getAffineDimExpr(dim));
      }
    }

    mlir::AffineMap inputMap = mlir::AffineMap::get(
        outputDimensions, 0, inputAffineExpressions, rewriter.getContext());

    llvm::SmallVector<mlir::Type, 1> resultTypes{outputType};
    auto ins = llvm::SmallVector<mlir::Value, 1>{input};
    auto outs = llvm::SmallVector<mlir::Value, 1>{emptyTensorOp.getResult()};
    llvm::SmallVector<mlir::AffineMap, 2> maps{
        inputMap,
        mlir::AffineMap::getMultiDimIdentityMap(outputDimensions,
                                                this->getContext()),
    };
    auto iteratorTypes = parallelIteratorType(outputDimensions);
    auto regionBuilder = [&](mlir::OpBuilder &nestedBuilder,
                             mlir::Location nestedLoc,
                             mlir::ValueRange blockArgs) {
      mlir::Value item = blockArgs[0];
      nestedBuilder.create<linalg::YieldOp>(location, item);
    };

    auto genericOp = rewriter.create<linalg::GenericOp>(
        location, resultTypes, ins, outs, maps, iteratorTypes, regionBuilder);

    if (broadcastOp->hasAttr("tile-sizes")) {
      genericOp->setAttr("tile-sizes", broadcastOp->getAttr("tile-sizes"));
    }

    rewriter.replaceOp(broadcastOp, genericOp.getResults());
    return mlir::success();
  };
};

// This operation should be used by the optimizer in multi-parameters, then
// removed. Its presence may indicate that mono-parameters might have been
// used. This patterns just hint for a potential fix.
struct ChangePartitionOpPattern
    : public mlir::OpRewritePattern<FHELinalg::ChangePartitionEintOp> {
  ChangePartitionOpPattern(mlir::MLIRContext *context)
      : mlir::OpRewritePattern<FHELinalg::ChangePartitionEintOp>(
            context, mlir::concretelang::DEFAULT_PATTERN_BENEFIT) {}

  mlir::LogicalResult
  matchAndRewrite(FHELinalg::ChangePartitionEintOp op,
                  mlir::PatternRewriter &rewriter) const override {
    op.emitError(llvm::Twine("change_partition shouldn't be present at this "
                             "level. Maybe you didn't use multi-parameters?"));
    return mlir::failure();
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
  target.addLegalDialect<mlir::arith::ArithDialect>();
  target.addIllegalOp<mlir::concretelang::FHELinalg::Dot>();
  target.addIllegalDialect<mlir::concretelang::FHELinalg::FHELinalgDialect>();

  target.addLegalOp<mlir::scf::ForallOp>();
  target.addLegalOp<mlir::scf::InParallelOp>();

  target.addIllegalOp<arith::IndexCastOp>();
  target.addDynamicallyLegalOp<arith::IndexCastOp>([&](arith::IndexCastOp op) {
    return !op.getOperand().getType().isa<mlir::RankedTensorType>() ||
           !op.getResult().getType().isa<mlir::RankedTensorType>();
  });

  target.addIllegalOp<arith::ExtSIOp>();
  target.addDynamicallyLegalOp<arith::ExtSIOp>([&](arith::ExtSIOp op) {
    return !op.getOperand().getType().isa<mlir::RankedTensorType>() ||
           !op.getResult().getType().isa<mlir::RankedTensorType>();
  });

  target.addDynamicallyLegalOp<
      mlir::concretelang::Optimizer::PartitionFrontierOp>(
      [&](mlir::concretelang::Optimizer::PartitionFrontierOp op) {
        return !op.getInput().getType().isa<mlir::RankedTensorType>() &&
               !op.getResult().getType().isa<mlir::RankedTensorType>();
      });

  mlir::RewritePatternSet patterns(&getContext());

  patterns.insert<DotToLinalgGeneric<mlir::concretelang::FHELinalg::Dot,
                                     mlir::concretelang::FHE::MulEintIntOp>>(
      &getContext(),
      [](mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type,
         mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintIntOp>(
            loc, type, arg0, arg1);
      },
      [](FHELinalg::Dot &dot, FHE::AddEintOp &add, FHE::MulEintIntOp &mul) {
        forwardOptimizerID(dot, add);
        forwardOptimizerID(dot, mul);
      });
  patterns.insert<DotToLinalgGeneric<mlir::concretelang::FHELinalg::DotEint,
                                     mlir::concretelang::FHE::MulEintOp>>(
      &getContext(),
      [](mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type,
         mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintOp>(loc, type,
                                                                  arg0, arg1);
      },
      [&](FHELinalg::DotEint &dot, FHE::AddEintOp &add, FHE::MulEintOp &mul) {
        // By convention the first elements of the vectors are nodes for the
        // multiplication and the last one is the node for the addition.
        mlir::Builder builder(&getContext());
        auto optimizerIdAttr =
            dot->getAttrOfType<mlir::DenseI32ArrayAttr>("TFHE.OId");
        if (optimizerIdAttr == nullptr)
          return;
        auto optimizerIds = optimizerIdAttr.asArrayRef();
        add->setAttr("TFHE.OId",
                     builder.getI32IntegerAttr(optimizerIds.back()));
        mul->setAttr("TFHE.OId",
                     builder.getDenseI32ArrayAttr(optimizerIds.drop_back()));
      });
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
  patterns.insert<
      FHELinalgOpToLinalgGeneric<mlir::concretelang::FHELinalg::MulEintOp,
                                 mlir::concretelang::FHE::MulEintOp>>(
      &getContext());

  patterns.insert<
      FHELinalgUnaryOpToLinalgGeneric<mlir::concretelang::FHELinalg::LsbEintOp,
                                      mlir::concretelang::FHE::LsbEintOp>>(
      &getContext());
  patterns.insert<
      FHELinalgUnaryOpToLinalgGeneric<mlir::concretelang::FHELinalg::NegEintOp,
                                      mlir::concretelang::FHE::NegEintOp>>(
      &getContext());

  patterns.insert<
      FHELinalgUnaryOpToLinalgGeneric<mlir::concretelang::FHELinalg::ToSignedOp,
                                      mlir::concretelang::FHE::ToSignedOp>>(
      &getContext());

  patterns.insert<FHELinalgUnaryOpToLinalgGeneric<
      mlir::concretelang::FHELinalg::ToUnsignedOp,
      mlir::concretelang::FHE::ToUnsignedOp>>(&getContext());

  patterns.insert<
      FHELinalgUnaryOpToLinalgGeneric<mlir::concretelang::FHELinalg::RoundOp,
                                      mlir::concretelang::FHE::RoundEintOp>>(
      &getContext());

  patterns.insert<FHELinalgUnaryOpToLinalgGeneric<
      mlir::concretelang::FHELinalg::ReinterpretPrecisionEintOp,
      mlir::concretelang::FHE::ReinterpretPrecisionEintOp>>(&getContext());
  patterns.insert<FHELinalgApplyLookupTableToLinalgGeneric>(&getContext());
  patterns.insert<FHELinalgMatmulToLinalgGeneric<
      mlir::concretelang::FHELinalg::MatMulEintIntOp,
      mlir::concretelang::FHE::MulEintIntOp>>(
      &getContext(),
      [](mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type,
         mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintIntOp>(
            loc, type, arg0, arg1);
      },
      [](FHELinalg::MatMulEintIntOp &dot, FHE::AddEintOp &add,
         FHE::MulEintIntOp &mul) {
        forwardOptimizerID(dot, add);
        forwardOptimizerID(dot, mul);
      });
  patterns.insert<FHELinalgMatmulToLinalgGeneric<
      mlir::concretelang::FHELinalg::MatMulIntEintOp,
      mlir::concretelang::FHE::MulEintIntOp>>(
      &getContext(),
      [](mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type,
         mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintIntOp>(
            loc, type, arg1, arg0);
      },
      [](FHELinalg::MatMulIntEintOp &dot, FHE::AddEintOp &add,
         FHE::MulEintIntOp &mul) {
        forwardOptimizerID(dot, add);
        forwardOptimizerID(dot, mul);
      });
  patterns.insert<FHELinalgMatmulToLinalgGeneric<
      mlir::concretelang::FHELinalg::MatMulEintEintOp,
      mlir::concretelang::FHE::MulEintOp>>(
      &getContext(),
      [](mlir::OpBuilder &builder, mlir::Location loc, mlir::Type type,
         mlir::Value arg0, mlir::Value arg1) {
        return builder.create<mlir::concretelang::FHE::MulEintOp>(loc, type,
                                                                  arg1, arg0);
      },
      [&](FHELinalg::MatMulEintEintOp &dot, FHE::AddEintOp &add,
          FHE::MulEintOp &mul) {
        // By convention the first elements of the vectors are nodes for the
        // multiplication and the last one is the node for the addition.
        mlir::Builder builder(&getContext());
        auto optimizerIdAttr =
            dot->getAttrOfType<mlir::DenseI32ArrayAttr>("TFHE.OId");
        if (optimizerIdAttr == nullptr)
          return;
        auto optimizerIds = optimizerIdAttr.asArrayRef();
        add->setAttr("TFHE.OId",
                     builder.getI32IntegerAttr(optimizerIds.back()));
        mul->setAttr("TFHE.OId",
                     builder.getDenseI32ArrayAttr(optimizerIds.drop_back()));
      });
  patterns.insert<FHELinalgApplyMultiLookupTableToLinalgGeneric>(&getContext());
  patterns.insert<FHELinalgApplyMappedLookupTableToLinalgGeneric>(
      &getContext());
  patterns.insert<SumToLinalgGeneric>(&getContext());
  patterns.insert<ConcatRewritePattern>(&getContext());
  patterns.insert<FHELinalgConv2dToLinalgConv2d>(&getContext());
  patterns.insert<FHELinalgMaxpool2dToLinalgMaxpool2d>(&getContext());
  patterns.insert<TransposeToLinalgGeneric>(&getContext());
  patterns.insert<FromElementToTensorFromElements>(&getContext());
  patterns.insert<TensorPartitionFrontierOpToLinalgGeneric>(&getContext());
  patterns.insert<FancyIndexToTensorGenerate>(&getContext());
  patterns.insert<FancyAssignToSfcForall>(&getContext());
  patterns.insert<ChangePartitionOpPattern>(&getContext());

  patterns.insert<TensorMapToLinalgGeneric<arith::IndexCastOp>>(&getContext());
  patterns.insert<TensorMapToLinalgGeneric<arith::ExtSIOp>>(&getContext());

  patterns.insert<BroadcastToLinalgGeneric>(&getContext());

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

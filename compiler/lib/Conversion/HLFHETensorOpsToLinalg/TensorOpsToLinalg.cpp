#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"
#include <iostream>

#include "zamalang/Conversion/Passes.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h"

struct DotToLinalgGeneric : public ::mlir::RewritePattern {
  DotToLinalgGeneric(::mlir::MLIRContext *context)
      : ::mlir::RewritePattern("HLFHE.dot_eint_int", 1, context,
                               {"linalg.generic"}) {}

  // This rewrite pattern transforms any instance of
  // `HLFHE.dot_eint_int` to an instance of `linalg.generic` with an
  // appropriate region using `HLFHE.mul_eint_int` and
  // `HLFHE.add_eint` operations, an appropriate specification for the
  // iteration dimensions and appropriate operaztions managing the
  // accumulator of `linalg.generic`.
  //
  // Example:
  //
  //   %o = "HLFHE.dot_eint_int"(%arg0, %arg1) :
  //     (tensor<4x!HLFHE.eint<0>>,
  //      tensor<4xi32>) -> (!HLFHE.eint<0>)
  //
  // becomes:
  //
  //   %0 = "HLFHE.zero"() : () -> !HLFHE.eint<0>
  //   %1 = tensor.from_elements %0 : tensor<1x!HLFHE.eint<0>>
  //   %2 = linalg.generic {
  //          indexing_maps = [#map0, #map0, #map1],
  //          iterator_types = ["reduction"]
  //        }
  //        ins(%arg0, %arg1 : tensor<2x!HLFHE.eint<0>>, tensor<2xi32>)
  //        outs(%1 : tensor<1x!HLFHE.eint<0>>) {
  //          ^bb0(%arg2: !HLFHE.eint<0>, %arg3: i32, %arg4: !HLFHE.eint<0>):
  //            %4 = "HLFHE.mul_eint_int"(%arg2, %arg3) :
  //                    (!HLFHE.eint<0>, i32) -> !HLFHE.eint<0>
  //
  //            %5 = "HLFHE.add_eint"(%4, %arg4) :
  //                    (!HLFHE.eint<0>, !HLFHE.eint<0>) -> !HLFHE.eint<0>
  //
  //            linalg.yield %5 : !HLFHE.eint<0>
  //        } -> tensor<1x!HLFHE.eint<0>>
  //
  //   %c0 = constant 0 : index
  //   %o = tensor.extract %2[%c0] : tensor<1x!HLFHE.eint<0>>
  //
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::zamalang::HLFHE::Dot &&dotOp =
        ::llvm::dyn_cast_or_null<::mlir::zamalang::HLFHE::Dot>(op0);

    // Zero value to initialize accumulator
    mlir::Value zeroCst = rewriter.create<mlir::zamalang::HLFHE::ZeroEintOp>(
        dotOp.getLoc(),
        dotOp.lhs().getType().cast<mlir::ShapedType>().getElementType());

    // Create one-dimensional accumulator with a single element
    // (`tensor.from_elements` does not allow for the creation of 0d
    // tensors)
    mlir::tensor::FromElementsOp feOp =
        rewriter.create<mlir::tensor::FromElementsOp>(dotOp.getLoc(), zeroCst);

    mlir::Value accu = feOp.getResult();

    // Create `linalg.generic` op
    llvm::SmallVector<mlir::Type, 1> resTypes{accu.getType()};
    llvm::SmallVector<mlir::Value, 2> ins{dotOp.lhs(), dotOp.rhs()};
    llvm::SmallVector<mlir::Value, 1> outs{accu};
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
      mlir::zamalang::HLFHE::MulEintIntOp mul =
          nestedBuilder.create<mlir::zamalang::HLFHE::MulEintIntOp>(
              dotOp.getLoc(), blockArgs[0], blockArgs[1]);
      mlir::zamalang::HLFHE::AddEintOp add =
          nestedBuilder.create<mlir::zamalang::HLFHE::AddEintOp>(
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

    rewriter.replaceOp(op0, {res});

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
  affineExprs.reserve(resultShape.size());
  size_t deltaNumDim = resultShape.size() - operandShape.size();
  for (auto i = 0; i < operandShape.size(); i++) {
    if (operandShape[i] == 1) {
      affineExprs.push_back(rewriter.getAffineConstantExpr(0));
    } else {
      affineExprs.push_back(rewriter.getAffineDimExpr(i + deltaNumDim));
    }
  }
  return mlir::AffineMap::get(resultShape.size(), 0, affineExprs,
                              rewriter.getContext());
}

// This template rewrite pattern transforms any instance of
// operators `HLFHELinalgOp` that implements the broadasting rules to an
// instance of `linalg.generic` with an appropriate region using `HLFHEOp`
// operation, an appropriate specification for the iteration dimensions and
// appropriate operaztions managing the accumulator of `linalg.generic`.
//
// Example:
//
// %res = HLFHELinalg.op(%lhs, %rhs):
// (tensor<D$Ax...xD1x!HLFHE.eint<p>>, tensor<D$B'x...xD1'xT>)
//    -> tensor<DR"x...xD1"x!HLFHE.eint<p>>
//
// becomes:
//
// #maps_0 = [
//    affine_map<(a$R", ..., a$A, ..., a1) ->
//        (dim(lhs, $A) == 1 ? 0 : a$A,..., dim(lhs, 1) == 1 ? 0 : a1)>,
//    affine_map<(a$R", ..., a1) ->
//        (dim(rhs, $B') == 1 ? 0 : a$B', ..., dim(rhs, 1) == 1 ? 0 : a1)>,
//    affine_map<(a$R", ..., a1) -> (a$R", ..., a1)
// ]
// #attributes_0 {
//     indexing_maps = #maps_0,
//     iterator_types = ["parallel", ..., "parallel"], // $R" parallel
// }
// %init = linalg.init_tensor [DR",...,D1"]
//            : tensor<DR"x...xD1"x!HLFHE.eint<p>>
// %res = linalg.generic {
//     ins(%lhs, %rhs: tensor<DAx...xD1x!HLFHE.eint<p>>,tensor<DB'x...xD1'xT>)
//     outs(%init : tensor<DR"x...xD1"x!HLFHE.eint<p>>)
//     {
//         ^bb0(%arg0: !HLFHE.eint<p>, %arg1: T):
//             %0 = HLFHE.op(%arg0, %arg1): !HLFHE.eint<p>, T ->
//             !HLFHE.eint<p>
//         linalg.yield %0 : !HLFHE.eint<p>
//     }
// }
//
template <typename HLFHELinalgOp, typename HLFHEOp>
struct HLFHELinalgOpToLinalgGeneric
    : public mlir::OpRewritePattern<HLFHELinalgOp> {
  HLFHELinalgOpToLinalgGeneric(::mlir::MLIRContext *context,
                               mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<HLFHELinalgOp>(context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(HLFHELinalgOp linalgOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)linalgOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType lhsTy =
        ((mlir::Type)linalgOp.lhs().getType()).cast<mlir::RankedTensorType>();
    mlir::RankedTensorType rhsTy =
        ((mlir::Type)linalgOp.rhs().getType()).cast<mlir::RankedTensorType>();
    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<mlir::linalg::InitTensorOp>(
        linalgOp.getLoc(), resultTy.getShape(), resultTy.getElementType());

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
      HLFHEOp hlfheOp = nestedBuilder.create<HLFHEOp>(
          linalgOp.getLoc(), blockArgs[0], blockArgs[1]);

      nestedBuilder.create<mlir::linalg::YieldOp>(linalgOp.getLoc(),
                                                  hlfheOp.getResult());
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

// This template rewrite pattern transforms any instance of
// operators `HLFHELinalg.apply_lookup_table` that implements the broadasting
// rules to an instance of `linalg.generic` with an appropriate region using
// `HLFHE.apply_lookup_table` operation, an appropriate specification for the
// iteration dimensions and appropriate operaztions managing the accumulator of
// `linalg.generic`.
//
// Example:
//
// HLFHELinalg.apply_lookup_table(%t, %lut):
//  tensor<DNx...xD1x!HLFHE.eint<p>>, tensor<DAxi64>
//      -> tensor<DNx...xD1x!HLFHE.eint<p'>>
//
// becomes:
//
// #maps_0 = [
//    affine_map<(aN, ..., a1) -> (aN, ..., a1)>,
//    affine_map<(aN, ..., a1) -> (aN, ..., a1)>
// ]
// #attributes_0 {
//     indexing_maps = #maps_0,
//     iterator_types = ["parallel",..],//N parallel
// }
// %init = linalg.init_tensor [DN,...,D1]
//            : tensor<DNx...xD1x!HLFHE.eint<p'>>
// %res = linalg.generic {
//     ins(%t: tensor<DNx...xD1x!HLFHE.eint<p>>)
//     outs(%init : tensor<DNx...xD1x!HLFHE.eint<p'>>)
//     {
//         ^bb0(%arg0: !HLFHE.eint<p>):
//             %0 = HLFHE.apply_lookup_table(%arg0, %lut): !HLFHE.eint<p>,
//             tensor<4xi64> -> !HLFHE.eint<p'>
//         linalg.yield %0 : !HLFHE.eint<p'>
//     }
// }
//
struct HLFHELinalgApplyLookupTableToLinalgGeneric
    : public mlir::OpRewritePattern<
          mlir::zamalang::HLFHELinalg::ApplyLookupTableEintOp> {
  HLFHELinalgApplyLookupTableToLinalgGeneric(::mlir::MLIRContext *context,
                                             mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<
            mlir::zamalang::HLFHELinalg::ApplyLookupTableEintOp>(context,
                                                                 benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::zamalang::HLFHELinalg::ApplyLookupTableEintOp lutOp,
                  ::mlir::PatternRewriter &rewriter) const override {
    mlir::RankedTensorType resultTy =
        ((mlir::Type)lutOp->getResult(0).getType())
            .cast<mlir::RankedTensorType>();
    mlir::RankedTensorType tTy =
        ((mlir::Type)lutOp.t().getType()).cast<mlir::RankedTensorType>();

    //  linalg.init_tensor for initial value
    mlir::Value init = rewriter.create<mlir::linalg::InitTensorOp>(
        lutOp.getLoc(), resultTy.getShape(), resultTy.getElementType());

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
      mlir::zamalang::HLFHE::ApplyLookupTableEintOp hlfheOp =
          nestedBuilder.create<mlir::zamalang::HLFHE::ApplyLookupTableEintOp>(
              lutOp.getLoc(), resultTy.getElementType(), blockArgs[0],
              lutOp.lut());

      nestedBuilder.create<mlir::linalg::YieldOp>(lutOp.getLoc(),
                                                  hlfheOp.getResult());
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

namespace {
struct HLFHETensorOpsToLinalg
    : public HLFHETensorOpsToLinalgBase<HLFHETensorOpsToLinalg> {

  void runOnFunction() final;
};

void HLFHETensorOpsToLinalg::runOnFunction() {
  mlir::FuncOp function = this->getFunction();

  mlir::ConversionTarget target(getContext());

  target.addLegalDialect<mlir::linalg::LinalgDialect>();
  target.addLegalDialect<mlir::StandardOpsDialect>();
  target.addLegalDialect<mlir::memref::MemRefDialect>();
  target.addLegalDialect<mlir::zamalang::HLFHE::HLFHEDialect>();
  target.addLegalDialect<mlir::tensor::TensorDialect>();
  target.addLegalDialect<mlir::arith::ArithmeticDialect>();
  target.addIllegalOp<mlir::zamalang::HLFHE::Dot>();
  target.addIllegalDialect<mlir::zamalang::HLFHELinalg::HLFHELinalgDialect>();

  mlir::OwningRewritePatternList patterns(&getContext());
  patterns.insert<DotToLinalgGeneric>(&getContext());
  patterns.insert<
      HLFHELinalgOpToLinalgGeneric<mlir::zamalang::HLFHELinalg::AddEintOp,
                                   mlir::zamalang::HLFHE::AddEintOp>>(
      &getContext());
  patterns.insert<
      HLFHELinalgOpToLinalgGeneric<mlir::zamalang::HLFHELinalg::AddEintIntOp,
                                   mlir::zamalang::HLFHE::AddEintIntOp>>(
      &getContext());
  patterns.insert<
      HLFHELinalgOpToLinalgGeneric<mlir::zamalang::HLFHELinalg::SubIntEintOp,
                                   mlir::zamalang::HLFHE::SubIntEintOp>>(
      &getContext());
  patterns.insert<
      HLFHELinalgOpToLinalgGeneric<mlir::zamalang::HLFHELinalg::MulEintIntOp,
                                   mlir::zamalang::HLFHE::MulEintIntOp>>(
      &getContext());
  patterns.insert<HLFHELinalgApplyLookupTableToLinalgGeneric>(&getContext());

  if (mlir::applyPartialConversion(function, target, std::move(patterns))
          .failed())
    this->signalPassFailure();
}

} // namespace

namespace mlir {
namespace zamalang {
std::unique_ptr<mlir::FunctionPass> createConvertHLFHETensorOpsToLinalg() {
  return std::make_unique<HLFHETensorOpsToLinalg>();
}
} // namespace zamalang
} // namespace mlir

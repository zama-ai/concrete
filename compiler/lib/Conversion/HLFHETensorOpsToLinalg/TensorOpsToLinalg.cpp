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
    mlir::Value zeroCst = rewriter.create<mlir::zamalang::HLFHE::ZeroOp>(
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
        rewriter.create<mlir::ConstantIndexOp>(dotOp.getLoc(), 0);
    llvm::SmallVector<mlir::Value, 1> indexes{idx0};
    mlir::Value res = rewriter.create<mlir::tensor::ExtractOp>(
        dotOp.getLoc(), gop.getResult(0), indexes);

    rewriter.replaceOp(op0, {res});

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
  target.addIllegalOp<mlir::zamalang::HLFHE::Dot>();

  mlir::OwningRewritePatternList patterns(&getContext());
  patterns.insert<DotToLinalgGeneric>(&getContext());

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

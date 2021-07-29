#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
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
  // `HLFHE.add_eint` operations and an appropriate specification for
  // the iteration dimensions.
  //
  // Example:
  //
  //   "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
  //     (memref<?x!HLFHE.eint<0>>,
  //      memref<?xi32>,
  //      memref<!HLFHE.eint<0>>) -> ()
  //
  // becomes:
  //
  //   linalg.generic {
  //     indexing_maps = [affine_map<(d0) -> (d0)>,
  //                      affine_map<(d0) -> (d0)>,
  //                      affine_map<(d0) -> ()>],
  //     iterator_types = ["reduction"]
  //   } ins(%arg0, %arg1 : memref<?x!HLFHE.eint<0>>, memref<?xi32>)
  //     outs(%arg2: memref<!HLFHE.eint<0>>)
  //   {
  //     ^bb0(%arg3: !HLFHE.eint<0>, %arg4: i32, %arg5: !HLFHE.eint<0>):
  //       %0 = "HLFHE.mul_eint_int"(%arg3, %arg4) : (!HLFHE.eint<0>, i32) ->
  //       !HLFHE.eint<0> %1 = "HLFHE.add_eint"(%0, %arg5) : (!HLFHE.eint<0>,
  //       !HLFHE.eint<0>) -> !HLFHE.eint<0> linalg.yield %1 : !HLFHE.eint<0>
  //   }
  //
  ::mlir::LogicalResult
  matchAndRewrite(::mlir::Operation *op0,
                  ::mlir::PatternRewriter &rewriter) const override {
    ::mlir::zamalang::HLFHE::Dot &&dotOp =
        ::llvm::dyn_cast_or_null<::mlir::zamalang::HLFHE::Dot>(op0);

    mlir::TypeRange resTypes{};
    llvm::SmallVector<mlir::Value, 2> ins{dotOp.lhs(), dotOp.rhs()};
    llvm::SmallVector<mlir::Value, 1> outs{dotOp.out()};

    llvm::SmallVector<mlir::AffineMap, 3> maps{
        mlir::AffineMap::getMultiDimIdentityMap(1, this->getContext()),
        mlir::AffineMap::getMultiDimIdentityMap(1, this->getContext()),
        mlir::AffineMap::get(1, 0, this->getContext())};

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

    rewriter.replaceOp(op0, {gop.getODSResults(0)});

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

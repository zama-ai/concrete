// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"

char move_bsk_to_gpu[] = "move_bsk_to_gpu";
char free_from_gpu[] = "free_from_gpu";

/// \brief Rewrites `BConcrete.move_bsk_to_gpu` into a CAPI call to
/// `move_bsk_to_gpu`
///
/// Also insert the forward declaration of `move_bsk_to_gpu`
struct MoveBskOpPattern : public mlir::OpRewritePattern<
                              mlir::concretelang::BConcrete::MoveBskToGPUOp> {
  MoveBskOpPattern(::mlir::MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::BConcrete::MoveBskToGPUOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::BConcrete::MoveBskToGPUOp moveBskOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto ctx = getContextArgument(moveBskOp);

    mlir::SmallVector<mlir::Value> operands{ctx};

    // Insert forward declaration of the function
    auto contextType =
        mlir::concretelang::Concrete::ContextType::get(rewriter.getContext());
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(), {contextType},
        {mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type())});
    if (insertForwardDeclaration(moveBskOp, rewriter, move_bsk_to_gpu, funcType)
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        moveBskOp, move_bsk_to_gpu, moveBskOp.getResult().getType(), operands);

    return ::mlir::success();
  };
};

/// \brief Rewrites `BConcrete.free_bsk_from_gpu` into a CAPI call to
/// `free_from_gpu`
///
/// Also insert the forward declaration of `free_from_gpu`
struct FreeBskOpPattern : public mlir::OpRewritePattern<
                              mlir::concretelang::BConcrete::FreeBskFromGPUOp> {
  FreeBskOpPattern(::mlir::MLIRContext *context,
                   mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<
            mlir::concretelang::BConcrete::FreeBskFromGPUOp>(context, benefit) {
  }

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::BConcrete::FreeBskFromGPUOp freeBskOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    mlir::SmallVector<mlir::Value> operands{freeBskOp.bsk()};

    // Insert forward declaration of the function
    auto funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {mlir::LLVM::LLVMPointerType::get(rewriter.getI64Type())}, {});
    if (insertForwardDeclaration(freeBskOp, rewriter, free_from_gpu, funcType)
            .failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<mlir::func::CallOp>(
        freeBskOp, free_from_gpu, mlir::TypeRange({}), operands);

    return ::mlir::success();
  };
};

namespace {
struct BConcreteToCAPIPass : public BConcreteToCAPIBase<BConcreteToCAPIPass> {
  void runOnOperation() final;
};
} // namespace

void BConcreteToCAPIPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());

  target.addIllegalOp<mlir::concretelang::BConcrete::MoveBskToGPUOp>();
  target.addLegalDialect<mlir::func::FuncDialect>();

  patterns.insert<MoveBskOpPattern>(&getContext());
  patterns.insert<FreeBskOpPattern>(&getContext());

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertBConcreteToCAPIPass() {
  return std::make_unique<BConcreteToCAPIPass>();
}
} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"

/// This rewrite pattern transforms any instance of `Concrete.bootstrap_lwe`
/// into `Concrete.bootstrap_lwe_gpu`. It also inserts operations to allocate
/// memory, copy bsk into GPU, and free memory after bootstrapping.
struct BstOpPattern : public mlir::OpRewritePattern<
                          mlir::concretelang::Concrete::BootstrapLweOp> {
  BstOpPattern(::mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::concretelang::Concrete::BootstrapLweOp>(
            context, benefit) {}

  ::mlir::LogicalResult
  matchAndRewrite(mlir::concretelang::Concrete::BootstrapLweOp bstOp,
                  ::mlir::PatternRewriter &rewriter) const override {

    auto baselog = bstOp.baseLog();
    auto level = bstOp.level();
    mlir::Value ct = bstOp.input_ciphertext();

    auto ctType =
        ct.getType().cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto inputLweDim = ctType.getDimension();

    auto outType = bstOp.getResult()
                       .getType()
                       .cast<mlir::concretelang::Concrete::LweCiphertextType>();
    auto outputLweDim = outType.getDimension();

    // copy bsk into GPU
    mlir::Value bskGPU =
        rewriter
            .create<mlir::concretelang::Concrete::MoveBskToGPUOp>(
                bstOp.getLoc(), mlir::concretelang::Concrete::GPUBskType::get(
                                    rewriter.getContext()))
            .getResult();

    mlir::Value inputLweDimCst = rewriter.create<mlir::arith::ConstantIntOp>(
        bstOp.getLoc(), inputLweDim, 32);
    mlir::Value polySizeCst = rewriter.create<mlir::arith::ConstantIntOp>(
        bstOp.getLoc(), outputLweDim, 32);
    mlir::Value levelCst =
        rewriter.create<mlir::arith::ConstantIntOp>(bstOp.getLoc(), level, 32);
    mlir::Value baselogCst = rewriter.create<mlir::arith::ConstantIntOp>(
        bstOp.getLoc(), baselog, 32);

    mlir::Type tableType =
        mlir::RankedTensorType::get({4}, rewriter.getI64Type());
    mlir::Value tableCst = rewriter.create<mlir::arith::ConstantOp>(
        bstOp.getLoc(),
        mlir::DenseIntElementsAttr::get(
            tableType, {llvm::APInt(64, 0), llvm::APInt(64, 0),
                        llvm::APInt(64, 0), llvm::APInt(64, 0)}));

    rewriter
        .replaceOpWithNewOp<mlir::concretelang::Concrete::BootstrapLweGPUOp>(
            bstOp, outType, ct, tableCst, inputLweDimCst, polySizeCst, levelCst,
            baselogCst, bskGPU);

    // free bsk memory from GPU
    rewriter.create<mlir::concretelang::Concrete::FreeBskFromGPUOp>(
        bstOp.getLoc(), bskGPU);

    return ::mlir::success();
  };
};

namespace {
struct ConcreteToGPUPass : public ConcreteToGPUBase<ConcreteToGPUPass> {
  void runOnOperation() final;
};
} // namespace

void ConcreteToGPUPass::runOnOperation() {
  auto op = this->getOperation();

  mlir::ConversionTarget target(getContext());
  mlir::RewritePatternSet patterns(&getContext());

  target.addLegalDialect<mlir::concretelang::Concrete::ConcreteDialect,
                         mlir::arith::ArithmeticDialect>();
  target.addIllegalOp<mlir::concretelang::Concrete::BootstrapLweOp>();

  patterns.insert<BstOpPattern>(&getContext());

  // Apply conversion
  if (mlir::applyPartialConversion(op, target, std::move(patterns)).failed()) {
    this->signalPassFailure();
  }
}

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertConcreteToGPUPass() {
  return std::make_unique<ConcreteToGPUPass>();
}
} // namespace concretelang
} // namespace mlir

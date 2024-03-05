// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/Transforms/Passes.h"

namespace {
struct AddRuntimeContextToFuncOpPattern
    : public mlir::OpRewritePattern<mlir::func::FuncOp> {
  AddRuntimeContextToFuncOpPattern(mlir::MLIRContext *context,
                                   mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<mlir::func::FuncOp>(context, benefit) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::func::FuncOp oldFuncOp,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::OpBuilder::InsertionGuard guard(rewriter);
    mlir::FunctionType oldFuncType = oldFuncOp.getFunctionType();

    // Add a Concrete.context to the function signature
    mlir::SmallVector<mlir::Type> newInputs(oldFuncType.getInputs().begin(),
                                            oldFuncType.getInputs().end());
    newInputs.push_back(
        rewriter.getType<mlir::concretelang::Concrete::ContextType>());
    mlir::FunctionType newFuncTy = rewriter.getType<mlir::FunctionType>(
        newInputs, oldFuncType.getResults());

    rewriter.updateRootInPlace(oldFuncOp,
                               [&] { oldFuncOp.setType(newFuncTy); });
    oldFuncOp.getBody().front().addArgument(
        rewriter.getType<mlir::concretelang::Concrete::ContextType>(),
        oldFuncOp.getLoc());

    return mlir::success();
  }

  /// Legal function are one that are private or has a Concrete.context as last
  /// arguments.
  static bool isLegal(mlir::func::FuncOp funcOp) {
    if (!funcOp.isPublic()) {
      return true;
    }

    return funcOp.getFunctionType().getNumInputs() >= 1 &&
           funcOp.getFunctionType()
               .getInputs()
               .back()
               .isa<mlir::concretelang::Concrete::ContextType>();
  }
};

namespace {
struct FunctionConstantOpConversion
    : public mlir::OpRewritePattern<mlir::func::ConstantOp> {
  FunctionConstantOpConversion(mlir::MLIRContext *ctx,
                               mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<mlir::func::ConstantOp>(ctx, benefit) {}
  ::mlir::LogicalResult
  matchAndRewrite(mlir::func::ConstantOp op,
                  mlir::PatternRewriter &rewriter) const override {
    auto symTab = mlir::SymbolTable::getNearestSymbolTable(op);
    auto funcOp = mlir::SymbolTable::lookupSymbolIn(symTab, op.getValue());
    assert(funcOp &&
           "Function symbol missing in symbol table for function constant op.");
    mlir::FunctionType funType = mlir::cast<mlir::func::FuncOp>(funcOp)
                                     .getFunctionType()
                                     .cast<mlir::FunctionType>();
    mlir::SmallVector<mlir::Type> newInputs(funType.getInputs().begin(),
                                            funType.getInputs().end());
    newInputs.push_back(
        rewriter.getType<mlir::concretelang::Concrete::ContextType>());
    mlir::FunctionType newFuncTy =
        rewriter.getType<mlir::FunctionType>(newInputs, funType.getResults());

    rewriter.updateRootInPlace(op, [&] { op.getResult().setType(newFuncTy); });
    return mlir::success();
  }
  static bool isLegal(mlir::func::ConstantOp fun) {
    auto symTab = mlir::SymbolTable::getNearestSymbolTable(fun);
    auto funcOp = mlir::SymbolTable::lookupSymbolIn(symTab, fun.getValue());
    assert(funcOp &&
           "Function symbol missing in symbol table for function constant op.");
    mlir::FunctionType funType = mlir::cast<mlir::func::FuncOp>(funcOp)
                                     .getFunctionType()
                                     .cast<mlir::FunctionType>();
    if ((AddRuntimeContextToFuncOpPattern::isLegal(
             mlir::cast<mlir::func::FuncOp>(funcOp)) &&
         fun.getType() == funType) ||
        fun.getType() != funType)
      return true;
    return false;
  }
};
} // namespace

struct AddRuntimeContextPass
    : public AddRuntimeContextBase<AddRuntimeContextPass> {
  void runOnOperation() final;
};

void AddRuntimeContextPass::runOnOperation() {
  mlir::ModuleOp op = getOperation();

  // First of all add the Concrete.context to the block arguments of function
  // that manipulates ciphertexts.
  {
    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    target.addDynamicallyLegalOp<mlir::func::FuncOp>(
        [&](mlir::func::FuncOp funcOp) {
          return AddRuntimeContextToFuncOpPattern::isLegal(funcOp);
        });
    target.addDynamicallyLegalOp<mlir::func::ConstantOp>(
        [&](mlir::func::ConstantOp op) {
          return FunctionConstantOpConversion::isLegal(op);
        });

    patterns.add<AddRuntimeContextToFuncOpPattern>(patterns.getContext());
    patterns.add<FunctionConstantOpConversion>(patterns.getContext());

    // Apply the conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
      return;
    }
  }
}
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createAddRuntimeContext() {
  return std::make_unique<AddRuntimeContextPass>();
}
} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"
#include "concretelang/Dialect/BConcrete/Transforms/Passes.h"

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
    // Create the new func
    mlir::func::FuncOp newFuncOp = rewriter.create<mlir::func::FuncOp>(
        oldFuncOp.getLoc(), oldFuncOp.getName(), newFuncTy);

    // Create the arguments of the new func
    mlir::Region &newFuncBody = newFuncOp.getBody();
    mlir::Block *newFuncEntryBlock = new mlir::Block();
    llvm::SmallVector<mlir::Location> locations(newFuncTy.getInputs().size(),
                                                oldFuncOp.getLoc());

    newFuncEntryBlock->addArguments(newFuncTy.getInputs(), locations);
    newFuncBody.push_back(newFuncEntryBlock);

    // Clone the old body to the new one
    mlir::BlockAndValueMapping map;
    for (auto arg : llvm::enumerate(oldFuncOp.getArguments())) {
      map.map(arg.value(), newFuncEntryBlock->getArgument(arg.index()));
    }
    for (auto &op : oldFuncOp.getBody().front()) {
      newFuncEntryBlock->push_back(op.clone(map));
    }
    rewriter.eraseOp(oldFuncOp);
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

    patterns.add<AddRuntimeContextToFuncOpPattern>(patterns.getContext());

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

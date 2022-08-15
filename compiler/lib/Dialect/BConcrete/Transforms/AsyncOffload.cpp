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

struct AsyncOffloadPass : public AsyncOffloadBase<AsyncOffloadPass> {
  void runOnOperation() final;
};

void AsyncOffloadPass::runOnOperation() {
  auto module = getOperation();
  std::vector<mlir::Operation *> ops;

  module.walk([&](mlir::concretelang::BConcrete::KeySwitchLweBufferOp op) {
    mlir::OpBuilder builder(op);
    mlir::Type futType =
        mlir::concretelang::RT::FutureType::get(op.getResult().getType());
    mlir::Value future = builder.create<
        mlir::concretelang::BConcrete::KeySwitchLweBufferAsyncOffloadOp>(
        op.getLoc(), mlir::TypeRange{futType}, op.getOperand(), op->getAttrs());

    assert(op.getResult().hasOneUse() &&
           "Single use assumed (for deallocation purposes - restriction can be "
           "lifted).");
    for (auto &use : op.getResult().getUses()) {
      builder.setInsertionPoint(use.getOwner());
      mlir::Value res =
          builder.create<mlir::concretelang::BConcrete::AwaitFutureOp>(
              use.getOwner()->getLoc(),
              mlir::TypeRange{op.getResult().getType()}, future);
      use.set(res);
    }
    ops.push_back(op);
  });
  module.walk([&](mlir::concretelang::BConcrete::BootstrapLweBufferOp op) {
    mlir::OpBuilder builder(op);
    mlir::Type futType =
        mlir::concretelang::RT::FutureType::get(op.getResult().getType());
    mlir::Value future = builder.create<
        mlir::concretelang::BConcrete::BootstrapLweBufferAsyncOffloadOp>(
        op.getLoc(), mlir::TypeRange{futType}, op.getOperands(),
        op->getAttrs());

    assert(op.getResult().hasOneUse() &&
           "Single use assumed (for deallocation purposes - restriction can be "
           "lifted).");
    for (auto &use : op.getResult().getUses()) {
      builder.setInsertionPoint(use.getOwner());
      mlir::Value res =
          builder.create<mlir::concretelang::BConcrete::AwaitFutureOp>(
              use.getOwner()->getLoc(),
              mlir::TypeRange{op.getResult().getType()}, future);
      use.set(res);
    }
    ops.push_back(op);
  });

  for (auto op : ops)
    op->erase();
}
} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createAsyncOffload() {
  return std::make_unique<AsyncOffloadPass>();
}
} // namespace concretelang
} // namespace mlir

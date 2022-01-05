// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include <iostream>

#include <concretelang/Dialect/RT/Analysis/Autopar.h>
#include <concretelang/Dialect/RT/IR/RTDialect.h>
#include <concretelang/Dialect/RT/IR/RTOps.h>
#include <concretelang/Dialect/RT/IR/RTTypes.h>

#include <concretelang/Conversion/Utils/GenericOpTypeConversionPattern.h>
#include <llvm/IR/Instructions.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/BlockAndValueMapping.h>
#include <mlir/IR/Builders.h>
#include <mlir/Transforms/Bufferize.h>
#include <mlir/Transforms/RegionUtils.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/RT/Analysis/Autopar.h.inc>

namespace mlir {
namespace concretelang {

namespace {
class BufferizeDataflowYieldOp
    : public OpConversionPattern<RT::DataflowYieldOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(RT::DataflowYieldOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    RT::DataflowYieldOp::Adaptor transformed(operands);
    rewriter.replaceOpWithNewOp<RT::DataflowYieldOp>(op, mlir::TypeRange(),
                                                     transformed.getOperands());
    return success();
  }
};
} // namespace

namespace {
class BufferizeDataflowTaskOp : public OpConversionPattern<RT::DataflowTaskOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(RT::DataflowTaskOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    RT::DataflowTaskOp::Adaptor transformed(operands);
    mlir::OpBuilder::InsertionGuard guard(rewriter);

    SmallVector<Type> newResults;
    (void)getTypeConverter()->convertTypes(op.getResultTypes(), newResults);
    auto newop = rewriter.create<RT::DataflowTaskOp>(op.getLoc(), newResults,
                                                     transformed.getOperands());
    // We cannot clone here as cloned ops must be legalized (so this
    // would break on the YieldOp).  Instead use mergeBlocks which
    // moves the ops instead of cloning.
    rewriter.mergeBlocks(op.getBody(), newop.getBody(),
                         newop.getBody()->getArguments());
    // Because of previous bufferization there are buffer cast ops
    // that have been generated for the previously tensor results of
    // some tasks.  These cannot just be replaced directly as the
    // task's results would still be live.
    for (auto res : llvm::enumerate(op.getResults())) {
      // If this result is getting bufferized ...
      if (res.value().getType() !=
          getTypeConverter()->convertType(res.value().getType())) {
        for (auto &use : llvm::make_early_inc_range(res.value().getUses())) {
          // ... and its uses are in `BufferCastOp`s, then we
          // replace further uses of the buffer cast.
          if (isa<mlir::memref::BufferCastOp>(use.getOwner())) {
            rewriter.replaceOp(use.getOwner(), {newop.getResult(res.index())});
          }
        }
      }
    }
    rewriter.replaceOp(op, {newop.getResults()});
    return success();
  }
};
} // namespace

void populateRTBufferizePatterns(BufferizeTypeConverter &typeConverter,
                                 RewritePatternSet &patterns) {
  patterns.add<BufferizeDataflowYieldOp, BufferizeDataflowTaskOp>(
      typeConverter, patterns.getContext());
}

namespace {
// For documentation see Autopar.td
struct BufferizeDataflowTaskOpsPass
    : public BufferizeDataflowTaskOpsBase<BufferizeDataflowTaskOpsPass> {

  void runOnOperation() override {
    auto module = getOperation();
    auto *context = &getContext();
    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

    populateRTBufferizePatterns(typeConverter, patterns);

    // Forbid all RT ops that still use/return tensors
    target.addDynamicallyLegalDialect<RT::RTDialect>(
        [&](Operation *op) { return typeConverter.isLegal(op); });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }

  BufferizeDataflowTaskOpsPass(bool debug) : debug(debug){};

protected:
  bool debug;
};
} // end anonymous namespace

std::unique_ptr<mlir::Pass> createBufferizeDataflowTaskOpsPass(bool debug) {
  return std::make_unique<BufferizeDataflowTaskOpsPass>(debug);
}
} // namespace concretelang
} // namespace mlir

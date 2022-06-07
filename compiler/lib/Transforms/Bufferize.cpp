// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/Transforms/Bufferize.h"
#include "concretelang/Transforms/Bufferize.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Operation.h"
#include "mlir/Transforms/Passes.h"
using namespace mlir;

namespace {
// In a finalizing bufferize conversion, we know that all tensors have been
// converted to memrefs, thus, this op becomes an identity.
class BufferizeTensorStoreOp
    : public OpConversionPattern<memref::TensorStoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(memref::TensorStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<memref::CopyOp>(op, op.tensor(), op.memref());
    return success();
  }
};
} // namespace

void populatePatterns(BufferizeTypeConverter &typeConverter,
                      RewritePatternSet &patterns) {
  mlir::populateEliminateBufferizeMaterializationsPatterns(typeConverter,
                                                           patterns);
  patterns.add<BufferizeTensorStoreOp>(typeConverter, patterns.getContext());
}

namespace {
struct FinalizingBufferizePass
    : public FinalizingBufferizeBase<FinalizingBufferizePass> {
  using FinalizingBufferizeBase<
      FinalizingBufferizePass>::FinalizingBufferizeBase;

  void runOnFunction() override {
    auto func = getFunction();
    auto *context = &getContext();

    BufferizeTypeConverter typeConverter;
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);
    populatePatterns(typeConverter, patterns);

    // If all result types are legal, and all block arguments are legal (ensured
    // by func conversion above), then all types in the program are legal.
    //
    // We also check that the operand types are legal to avoid creating invalid
    // IR. For example, this prevents
    // populateEliminateBufferizeMaterializationsPatterns from updating the
    // types of the operands to a return op without updating the enclosing
    // function.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    target.addLegalOp<memref::CopyOp>();

    if (failed(applyFullConversion(func, target, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<FunctionPass>
mlir::concretelang::createFinalizingBufferizePass() {
  return std::make_unique<FinalizingBufferizePass>();
}
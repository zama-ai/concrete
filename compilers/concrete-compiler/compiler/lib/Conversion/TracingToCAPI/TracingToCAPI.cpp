// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include <mlir/IR/BuiltinTypeInterfaces.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>

#include "concretelang/Conversion/Passes.h"
#include "concretelang/Conversion/Tools.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferUtils.h"

namespace {

namespace Tracing = mlir::concretelang::Tracing;
namespace arith = mlir::arith;
namespace func = mlir::func;
namespace memref = mlir::memref;

char memref_trace_ciphertext[] = "memref_trace_ciphertext";
char memref_trace_plaintext[] = "memref_trace_plaintext";
char memref_trace_message[] = "memref_trace_message";

mlir::Type getDynamicMemrefWithUnknownOffset(mlir::RewriterBase &rewriter,
                                             size_t rank) {
  std::vector<int64_t> shape(rank, mlir::ShapedType::kDynamic);
  mlir::AffineExpr expr = rewriter.getAffineSymbolExpr(0);
  for (size_t i = 0; i < rank; i++) {
    expr = expr +
           (rewriter.getAffineDimExpr(i) * rewriter.getAffineSymbolExpr(i + 1));
  }
  return mlir::MemRefType::get(
      shape, rewriter.getI64Type(),
      mlir::AffineMap::get(rank, rank + 1, expr, rewriter.getContext()));
}

// Returns `memref.cast %0 : memref<...xAxT> to memref<...x?xT>`
mlir::Value getCastedMemRef(mlir::RewriterBase &rewriter, mlir::Value value) {
  mlir::Type valueType = value.getType();

  if (auto memrefTy = valueType.dyn_cast_or_null<mlir::MemRefType>()) {
    return rewriter.create<mlir::memref::CastOp>(
        value.getLoc(),
        getDynamicMemrefWithUnknownOffset(rewriter, memrefTy.getShape().size()),
        value);
  } else {
    return value;
  }
}

mlir::LogicalResult insertForwardDeclarationOfTheCAPI(
    mlir::Operation *op, mlir::RewriterBase &rewriter, char const *funcName) {

  auto memref1DType = getDynamicMemrefWithUnknownOffset(rewriter, 1);

  mlir::FunctionType funcType;

  if (funcName == memref_trace_ciphertext) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {memref1DType, mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()),
         rewriter.getI32Type(), rewriter.getI32Type()},
        {});
  } else if (funcName == memref_trace_plaintext) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {rewriter.getI64Type(), rewriter.getI64Type(),
         mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()),
         rewriter.getI32Type(), rewriter.getI32Type()},
        {});
  } else if (funcName == memref_trace_message) {
    funcType = mlir::FunctionType::get(
        rewriter.getContext(),
        {mlir::LLVM::LLVMPointerType::get(rewriter.getI8Type()),
         rewriter.getI32Type()},
        {});
  } else {
    op->emitError("unknwon external function") << funcName;
    return mlir::failure();
  }

  return insertForwardDeclaration(op, rewriter, funcName, funcType);
}

template <typename Op>
void addNoOperands(Op op, mlir::SmallVector<mlir::Value> &operands,
                   mlir::RewriterBase &rewriter) {}

template <typename Op, char const *callee>
struct TracingToCAPICallPattern : public mlir::OpRewritePattern<Op> {
  TracingToCAPICallPattern(
      ::mlir::MLIRContext *context,
      std::function<void(Op op, llvm::SmallVector<mlir::Value> &,
                         mlir::RewriterBase &)>
          addOperands = addNoOperands<Op>,
      mlir::PatternBenefit benefit = 1)
      : ::mlir::OpRewritePattern<Op>(context, benefit),
        addOperands(addOperands) {}

  ::mlir::LogicalResult
  matchAndRewrite(Op op, ::mlir::PatternRewriter &rewriter) const override {

    // Create the operands
    mlir::SmallVector<mlir::Value> operands;
    // For all tensor operand get the corresponding casted buffer
    for (auto &operand : op->getOpOperands()) {
      mlir::Type type = operand.get().getType();
      if (!type.isa<mlir::MemRefType>()) {
        operands.push_back(operand.get());
      } else {
        operands.push_back(getCastedMemRef(rewriter, operand.get()));
      }
    }

    // append additional argument
    addOperands(op, operands, rewriter);

    // Insert forward declaration of the function
    if (insertForwardDeclarationOfTheCAPI(op, rewriter, callee).failed()) {
      return mlir::failure();
    }

    rewriter.replaceOpWithNewOp<func::CallOp>(op, callee, mlir::TypeRange{},
                                              operands);

    return ::mlir::success();
  };

private:
  std::function<void(Op op, llvm::SmallVector<mlir::Value> &,
                     mlir::RewriterBase &)>
      addOperands;
};

void traceCiphertextAddOperands(Tracing::TraceCiphertextOp op,
                                mlir::SmallVector<mlir::Value> &operands,
                                mlir::RewriterBase &rewriter) {
  auto msg = op.getMsg().value_or("");
  auto nmsb = op.getNmsb().value_or(0);
  std::string msgName;
  std::stringstream stream;
  stream << rand();
  stream >> msgName;
  auto messageVal = mlir::LLVM::createGlobalString(
      op.getLoc(), rewriter, msgName, msg,
      mlir::LLVM::linkage::Linkage::Linkonce, false);
  operands.push_back(messageVal);
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), rewriter.getI32IntegerAttr(msg.size())));
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), rewriter.getI32IntegerAttr(nmsb)));
}

void tracePlaintextAddOperands(Tracing::TracePlaintextOp op,
                               mlir::SmallVector<mlir::Value> &operands,
                               mlir::RewriterBase &rewriter) {
  auto msg = op.getMsg().value_or("");
  auto nmsb = op.getNmsb().value_or(0);
  std::string msgName;
  std::stringstream stream;
  stream << rand();
  stream >> msgName;
  auto messageVal = mlir::LLVM::createGlobalString(
      op.getLoc(), rewriter, msgName, msg,
      mlir::LLVM::linkage::Linkage::Linkonce, false);
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), op->getAttr("input_width")));
  operands.push_back(messageVal);
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), rewriter.getI32IntegerAttr(msg.size())));
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), rewriter.getI32IntegerAttr(nmsb)));
}

void traceMessageAddOperands(Tracing::TraceMessageOp op,
                             mlir::SmallVector<mlir::Value> &operands,
                             mlir::RewriterBase &rewriter) {
  auto msg = op.getMsg().value_or("");
  std::string msgName;
  std::stringstream stream;
  stream << rand();
  stream >> msgName;
  auto messageVal = mlir::LLVM::createGlobalString(
      op.getLoc(), rewriter, msgName, msg,
      mlir::LLVM::linkage::Linkage::Linkonce, false);
  operands.push_back(messageVal);
  operands.push_back(rewriter.create<mlir::arith::ConstantOp>(
      op.getLoc(), rewriter.getI32IntegerAttr(msg.size())));
}

struct TracingToCAPIPass : public TracingToCAPIBase<TracingToCAPIPass> {

  TracingToCAPIPass() {}

  void runOnOperation() override {
    auto op = this->getOperation();

    mlir::ConversionTarget target(getContext());
    mlir::RewritePatternSet patterns(&getContext());

    // Mark ops from the target dialect as legal operations
    target.addLegalDialect<func::FuncDialect>();
    target.addLegalDialect<memref::MemRefDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<mlir::LLVM::LLVMDialect>();

    // Make sure that no ops from `Tracing` remain after the lowering
    target.addIllegalDialect<Tracing::TracingDialect>();

    // Add patterns to transform Tracing operators to CAPI call
    patterns.add<TracingToCAPICallPattern<Tracing::TraceCiphertextOp,
                                          memref_trace_ciphertext>>(
        &getContext(), traceCiphertextAddOperands);
    patterns.add<TracingToCAPICallPattern<Tracing::TracePlaintextOp,
                                          memref_trace_plaintext>>(
        &getContext(), tracePlaintextAddOperands);
    patterns.add<TracingToCAPICallPattern<Tracing::TraceMessageOp,
                                          memref_trace_message>>(
        &getContext(), traceMessageAddOperands);

    // Apply conversion
    if (mlir::applyPartialConversion(op, target, std::move(patterns))
            .failed()) {
      this->signalPassFailure();
    }
  }
};

} // namespace

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createConvertTracingToCAPIPass() {
  return std::make_unique<TracingToCAPIPass>();
}
} // namespace concretelang
} // namespace mlir

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_GENERICOPTYPECONVERSIONPATTERN_H_
#define CONCRETELANG_CONVERSION_GENERICOPTYPECONVERSIONPATTERN_H_

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/Support/LLVM.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace concretelang {

// Converts the type of all operands and the return type of `op` by
// invoking `convertType`
static inline void convertOperandAndResultTypes(
    mlir::PatternRewriter &rewriter, mlir::Operation *op,
    llvm::function_ref<mlir::Type(mlir::MLIRContext *, mlir::Type)>
        convertType) {
  rewriter.startRootUpdate(op);
  // Rewrite arguments
  {
    for (unsigned i = 0; i < op->getNumOperands(); i++) {
      auto operand = op->getOperand(i);
      mlir::Type type = convertType(rewriter.getContext(), operand.getType());
      if (type != mlir::Type()) {
        operand.setType(type);
      }
    }
  }
  // Rewrite results
  {
    for (unsigned i = 0; i < op->getNumResults(); i++) {
      auto result = op->getResult(i);
      mlir::Type type = convertType(rewriter.getContext(), result.getType());
      if (type != mlir::Type()) {
        result.setType(type);
      }
    }
  }

  rewriter.finalizeRootUpdate(op);
}

template <typename Op>
struct GenericTypeConverterPattern : public mlir::OpRewritePattern<Op> {
  GenericTypeConverterPattern(mlir::MLIRContext *context,
                              mlir::TypeConverter &converter,
                              mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<Op>(context, benefit), converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {
    auto newOp = rewriter.clone(*op);
    convertOperandAndResultTypes(rewriter, newOp,
                                 [&](mlir::MLIRContext *, mlir::Type t) {
                                   return converter.convertType(t);
                                 });
    rewriter.replaceOp(op, newOp->getResults());
    return mlir::success();
  }

private:
  mlir::TypeConverter &converter;
};

template <typename OldOp, typename NewOp>
struct GenericTypeAndOpConverterPattern : public mlir::OpRewritePattern<OldOp> {
  GenericTypeAndOpConverterPattern(mlir::MLIRContext *context,
                                   mlir::TypeConverter &converter,
                                   mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<OldOp>(context, benefit), converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(OldOp oldOp, mlir::PatternRewriter &rewriter) const override {
    // Rewrite results
    mlir::SmallVector<mlir::Type> resultTypes(oldOp->getNumResults());
    {
      for (unsigned i = 0; i < oldOp->getNumResults(); i++) {
        auto result = oldOp->getResult(i);
        resultTypes[i] = converter.convertType(result.getType());
      }
    }
    auto newOp = rewriter.replaceOpWithNewOp<NewOp>(
        oldOp, resultTypes, oldOp->getOperands(), oldOp->getAttrs());
    mlir::concretelang::convertOperandAndResultTypes(
        rewriter, newOp, [&](mlir::MLIRContext *, mlir::Type t) {
          return converter.convertType(t);
        });
    return mlir::success();
  }

private:
  mlir::TypeConverter &converter;
};
} // namespace concretelang
} // namespace mlir

#endif

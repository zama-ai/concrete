// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_GENERICOPTYPECONVERSIONPATTERN_H_
#define CONCRETELANG_CONVERSION_GENERICOPTYPECONVERSIONPATTERN_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace concretelang {
template <typename Op>
struct GenericTypeConverterPattern : public mlir::OpRewritePattern<Op> {
  GenericTypeConverterPattern(mlir::MLIRContext *context,
                              mlir::TypeConverter &converter,
                              mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<Op>(context, benefit), converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(Op op, mlir::PatternRewriter &rewriter) const override {

    rewriter.startRootUpdate(op);
    // Rewrite arguments
    {
      for (unsigned i = 0; i < op->getNumOperands(); i++) {
        auto operand = op->getOperand(i);
        mlir::Type type = converter.convertType(operand.getType());
        if (type != mlir::Type()) {
          operand.setType(type);
        }
      }
    }
    // Rewrite results
    {
      for (unsigned i = 0; i < op->getNumResults(); i++) {
        auto result = op->getResult(i);
        mlir::Type type = converter.convertType(result.getType());
        if (type != mlir::Type()) {
          result.setType(type);
        }
      }
    }
    rewriter.finalizeRootUpdate(op);
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
    rewriter.replaceOpWithNewOp<NewOp>(oldOp, resultTypes, oldOp->getOperands(),
                                       oldOp->getAttrs());
    return mlir::success();
  }

private:
  mlir::TypeConverter &converter;
};

template <typename Op>
void addDynamicallyLegalTypeOp(mlir::ConversionTarget &target,
                               mlir::TypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<Op>([&](Op op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
}

} // namespace concretelang
} // namespace mlir

#endif
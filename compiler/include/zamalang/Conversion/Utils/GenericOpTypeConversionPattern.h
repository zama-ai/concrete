#ifndef ZAMALANG_CONVERSION_GENERICOPTYPECONVERSIONPATTERN_H_
#define ZAMALANG_CONVERSION_GENERICOPTYPECONVERSIONPATTERN_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace zamalang {
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

template <typename Op>
void addDynamicallyLegalTypeOp(mlir::ConversionTarget &target,
                               mlir::TypeConverter &typeConverter) {
  target.addDynamicallyLegalOp<Op>([&](Op op) {
    return typeConverter.isLegal(op->getOperandTypes()) &&
           typeConverter.isLegal(op->getResultTypes());
  });
}

} // namespace zamalang
} // namespace mlir

#endif
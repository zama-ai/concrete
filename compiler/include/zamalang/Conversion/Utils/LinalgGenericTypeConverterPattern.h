#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/IR/PatternMatch.h"

/// LinalgGenericTypeConverterPattern is a rewrite pattern that convert types
/// `linalg.generic` operation, using a specific `typeConverter`
template <typename TypeConverter>
struct LinalgGenericTypeConverterPattern
    : public mlir::OpRewritePattern<mlir::linalg::GenericOp> {
  LinalgGenericTypeConverterPattern(mlir::MLIRContext *context,
                                    TypeConverter &converter,
                                    mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<mlir::linalg::GenericOp>(context, benefit),
        converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::linalg::GenericOp op,
                  mlir::PatternRewriter &rewriter) const override {

    rewriter.startRootUpdate(op);
    // Rewrite arguments
    {
      for (auto i = 0; i < op->getNumOperands(); i++) {
        auto operand = op->getOperand(i);
        mlir::Type type = converter.convertType(operand.getType());
        if (type != mlir::Type()) {
          operand.setType(type);
        }
      }
    }
    // Rewrite results
    {
      for (auto i = 0; i < op->getNumResults(); i++) {
        auto result = op->getResult(i);
        mlir::Type type = converter.convertType(result.getType());
        if (type != mlir::Type()) {
          result.setType(type);
        }
      }
    }
    // Rewrite block arguments
    mlir::Region &region = op->getRegion(0);
    mlir::Block *entry = &region.front();
    for (auto arg : entry->getArguments()) {
      mlir::Type type = converter.convertType(arg.getType());
      if (type != mlir::Type()) {
        arg.setType(type);
      }
    }
    rewriter.finalizeRootUpdate(op);
    return mlir::success();
  }

private:
  TypeConverter &converter;
};

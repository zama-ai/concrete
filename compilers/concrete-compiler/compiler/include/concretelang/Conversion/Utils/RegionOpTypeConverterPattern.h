// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/PatternMatch.h"

/// RegionOpTypeConverterPattern is a rewrite pattern that applies
/// `TypeConverter` to an instance of `OpWithRegion`, converting the
/// type of all operands, results and arguments of regions according
/// to the type converter.
template <typename OpWithRegion, typename TypeConverter>
struct RegionOpTypeConverterPattern
    : public mlir::OpRewritePattern<OpWithRegion> {
  RegionOpTypeConverterPattern(mlir::MLIRContext *context,
                               TypeConverter &converter,
                               mlir::PatternBenefit benefit = 100)
      : mlir::OpRewritePattern<OpWithRegion>(context, benefit),
        converter(converter) {}

  mlir::LogicalResult
  matchAndRewrite(OpWithRegion op,
                  mlir::PatternRewriter &rewriter) const override {
    auto doConvertType = [&](mlir::Value v) {
      mlir::Type type = converter.convertType(v.getType());

      if (type != mlir::Type())
        v.setType(type);
    };

    rewriter.startRootUpdate(op);
    llvm::for_each(op->getOperands(), doConvertType);
    llvm::for_each(op->getResults(), doConvertType);
    llvm::for_each(op->getRegions(), [&](mlir::Region &region) {
      llvm::for_each(region.front().getArguments(), doConvertType);
    });

    rewriter.finalizeRootUpdate(op);
    return mlir::success();
  }

private:
  TypeConverter &converter;
};

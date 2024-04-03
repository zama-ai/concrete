// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Conversion/Utils/Dialects/SCF.h"
#include "concretelang/Conversion/Utils/Utils.h"

namespace mlir {
namespace concretelang {
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::ForOp, false>::matchAndRewrite(
    scf::ForOp oldOp, mlir::OpConversionPattern<scf::ForOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::TypeConverter &typeConverter = *getTypeConverter();
  llvm::SmallVector<mlir::Type> convertedResultTypes;

  if (typeConverter.convertTypes(oldOp.getResultTypes(), convertedResultTypes)
          .failed()) {
    return mlir::failure();
  }

  convertOpWithBlocks(oldOp, adaptor.getOperands(), convertedResultTypes,
                      typeConverter, rewriter);

  return mlir::success();
}

//
// Specializations for ForallOp
//
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::ForallOp, false>::matchAndRewrite(
    scf::ForallOp oldOp,
    mlir::OpConversionPattern<scf::ForallOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {

  convertOpWithBlocks(oldOp, adaptor.getOperands(),
                      adaptor.getOutputs().getTypes(), *getTypeConverter(),
                      rewriter);

  return mlir::success();
}

//
// Specializations for InParallelOp
//
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::InParallelOp, false>::matchAndRewrite(
    scf::InParallelOp oldOp,
    mlir::OpConversionPattern<scf::InParallelOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  // Create new for loop with empty body, but converted iter args
  scf::InParallelOp newInParallelOp =
      rewriter.replaceOpWithNewOp<scf::InParallelOp>(oldOp);

  // Move operations from old for op to new one
  auto &newOperations = newInParallelOp.getBody()->getOperations();
  mlir::Block *oldBody = oldOp.getBody();

  newOperations.splice(newOperations.begin(), oldBody->getOperations(),
                       oldBody->begin(), oldBody->end());

  return mlir::success();
}

} // namespace concretelang
} // namespace mlir

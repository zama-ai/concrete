// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Conversion/Utils/Dialects/Tensor.h"
#include "concretelang/Conversion/Utils/Utils.h"

namespace mlir {
namespace concretelang {

//
// Specializations for CollapseShapeOp
//

// Specialization copying attributes not necessary, as the base
// template works correctly
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<tensor::CollapseShapeOp, false>::
    matchAndRewrite(
        tensor::CollapseShapeOp oldOp,
        mlir::OpConversionPattern<tensor::CollapseShapeOp>::OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const {
  mlir::SmallVector<mlir::Type> resultTypes = convertResultTypes(oldOp);
  rewriter.replaceOpWithNewOp<tensor::CollapseShapeOp>(
      oldOp, mlir::TypeRange{resultTypes}, adaptor.getSrc(),
      oldOp.getReassociation());

  return mlir::success();
}

//
// Specializations for FromElementsOp
//
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<mlir::tensor::FromElementsOp, false>::
    matchAndRewrite(
        tensor::FromElementsOp oldOp,
        mlir::OpConversionPattern<mlir::tensor::FromElementsOp>::OpAdaptor
            adaptor,
        mlir::ConversionPatternRewriter &rewriter) const {
  mlir::Type resultType = convertResultType(oldOp);
  rewriter.replaceOpWithNewOp<mlir::tensor::FromElementsOp>(
      oldOp, resultType, adaptor.getElements());

  return mlir::success();
}

//
// Specializations for ExpandShapeOp
//

// Specialization copying attributes not necessary, as the base
// template works correctly

template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<tensor::ExpandShapeOp, false>::
    matchAndRewrite(
        tensor::ExpandShapeOp oldOp,
        mlir::OpConversionPattern<tensor::ExpandShapeOp>::OpAdaptor adaptor,
        mlir::ConversionPatternRewriter &rewriter) const {
  mlir::SmallVector<mlir::Type> resultTypes = convertResultTypes(oldOp);
  rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
      oldOp, mlir::TypeRange{resultTypes}, adaptor.getSrc(),
      oldOp.getReassociation());

  return mlir::success();
}

template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<tensor::GenerateOp, true>::matchAndRewrite(
    tensor::GenerateOp oldOp,
    mlir::OpConversionPattern<tensor::GenerateOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const {
  mlir::SmallVector<mlir::Type> resultTypes = convertResultTypes(oldOp);

  convertOpWithBlocks(oldOp, adaptor.getOperands(), resultTypes,
                      *getTypeConverter(), rewriter);

  return mlir::success();
}

} // namespace concretelang
} // namespace mlir

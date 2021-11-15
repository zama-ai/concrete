#ifndef ZAMALANG_CONVERSION_TENSOROPTYPECONVERSIONPATTERN_H_
#define ZAMALANG_CONVERSION_TENSOROPTYPECONVERSIONPATTERN_H_

#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/Utils/GenericOpTypeConversionPattern.h"

namespace mlir {
namespace zamalang {

inline void
populateWithTensorTypeConverterPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::ConversionTarget &target,
                                        mlir::TypeConverter &typeConverter) {
  // ExtractOp
  patterns.add<GenericTypeConverterPattern<mlir::tensor::ExtractOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::tensor::ExtractOp>(target, typeConverter);
  // ExtractSliceOp
  patterns.add<GenericTypeConverterPattern<mlir::tensor::ExtractSliceOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::tensor::ExtractSliceOp>(target,
                                                          typeConverter);
  // InsertSliceOp
  patterns.add<GenericTypeConverterPattern<mlir::tensor::InsertSliceOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::tensor::InsertSliceOp>(target, typeConverter);

  // FromElementsOp
  patterns.add<GenericTypeConverterPattern<mlir::tensor::FromElementsOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::tensor::FromElementsOp>(target,
                                                          typeConverter);
  // TensorCollapseShapeOp
  patterns
      .add<GenericTypeConverterPattern<mlir::linalg::TensorCollapseShapeOp>>(
          patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::linalg::TensorCollapseShapeOp>(target,
                                                                 typeConverter);
  // TensorExpandShapeOp
  patterns.add<GenericTypeConverterPattern<mlir::linalg::TensorExpandShapeOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::linalg::TensorExpandShapeOp>(target,
                                                               typeConverter);
}
} // namespace zamalang
} // namespace mlir

#endif
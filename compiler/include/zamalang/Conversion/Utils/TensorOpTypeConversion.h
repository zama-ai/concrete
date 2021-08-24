#ifndef ZAMALANG_CONVERSION_TENSOROPTYPECONVERSIONPATTERN_H_
#define ZAMALANG_CONVERSION_TENSOROPTYPECONVERSIONPATTERN_H_

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "zamalang/Conversion/Utils/GenericOpTypeConversionPattern.h"

namespace mlir {
namespace zamalang {

inline void
populateWithTensorTypeConverterPatterns(mlir::RewritePatternSet &patterns,
                                        mlir::ConversionTarget &target,
                                        mlir::TypeConverter &typeConverter) {
  patterns.add<GenericTypeConverterPattern<mlir::tensor::ExtractOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::tensor::ExtractOp>(target, typeConverter);
  patterns.add<GenericTypeConverterPattern<mlir::tensor::FromElementsOp>>(
      patterns.getContext(), typeConverter);
  addDynamicallyLegalTypeOp<mlir::tensor::FromElementsOp>(target,
                                                          typeConverter);
}
} // namespace zamalang
} // namespace mlir

#endif
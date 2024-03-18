// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_UTILS_LEGALITY_H_
#define CONCRETELANG_CONVERSION_UTILS_LEGALITY_H_

#include <mlir/Transforms/DialectConversion.h>

namespace mlir {
namespace concretelang {

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

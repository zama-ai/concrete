// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_UTILS_DIALECTS_SCF_H_
#define CONCRETELANG_CONVERSION_UTILS_DIALECTS_SCF_H_

#include "concretelang/Conversion/Utils/ReinstantiatingOpTypeConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace concretelang {

//
// Specializations for ForOp
//

// Specialization copying attributes omitted
template <>
mlir::LogicalResult
TypeConvertingReinstantiationPattern<scf::ForOp, false>::matchAndRewrite(
    scf::ForOp oldOp, mlir::OpConversionPattern<scf::ForOp>::OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) const;

} // namespace concretelang
} // namespace mlir

#endif

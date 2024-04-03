// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_UTILS_H_
#define CONCRETELANG_CONVERSION_UTILS_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace concretelang {

mlir::Type getDynamicMemrefWithUnknownOffset(mlir::RewriterBase &rewriter,
                                             size_t rank);

// Returns `memref.cast %0 : memref<...xAxT> to memref<...x?xT>`
mlir::Value getCastedMemRef(mlir::RewriterBase &rewriter, mlir::Value value);

mlir::Value globalMemrefFromArrayAttr(mlir::RewriterBase &rewriter,
                                      mlir::Location loc,
                                      mlir::ArrayAttr arrAttr);

mlir::Operation *convertOpWithBlocks(mlir::Operation *op,
                                     mlir::ValueRange newOperands,
                                     mlir::TypeRange newResultTypes,
                                     mlir::TypeConverter &typeConverter,
                                     mlir::ConversionPatternRewriter &rewriter);

} // namespace concretelang
} // namespace mlir
#endif

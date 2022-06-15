// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/IR/PatternMatch.h"

mlir::LogicalResult insertForwardDeclaration(mlir::Operation *op,
                                             mlir::RewriterBase &rewriter,
                                             llvm::StringRef funcName,
                                             mlir::FunctionType funcType);

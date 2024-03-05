// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_FHETENSOROPSTOLINALG_PASS_H_
#define CONCRETELANG_CONVERSION_FHETENSOROPSTOLINALG_PASS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace concretelang {
/// Create a pass to convert `FHE` tensor operators to linal.generic
/// operators.
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createConvertFHETensorOpsToLinalg();
} // namespace concretelang
} // namespace mlir

#endif

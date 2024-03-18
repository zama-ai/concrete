// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONVERSION_EXTRACTSDFGOPS_PASS_H_
#define CONCRETELANG_CONVERSION_EXTRACTSDFGOPS_PASS_H_

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createExtractSDFGOpsPass(bool unroll);
} // namespace concretelang
} // namespace mlir

#endif

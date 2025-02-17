// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_GLWE_OPTIMIZATION_PASS_H
#define CONCRETELANG_GLWE_OPTIMIZATION_PASS_H

#include "mlir/Pass/Pass.h"

#include "concretelang/Dialect/GLWE/IR/GLWEDialect.h"

#define GEN_PASS_CLASSES
#include "concretelang/Dialect/GLWE/Transforms/Transforms.h.inc"

namespace mlir {
namespace concretelang {
namespace GLWE {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createInjectDefaultVariances();

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createPropagateVariances();
} // namespace GLWE
} // namespace concretelang
} // namespace mlir

#endif

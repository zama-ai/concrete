// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_CONCRETE_TRANSFORMS_PASSES_H_
#define CONCRETELANG_DIALECT_CONCRETE_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

#define GEN_PASS_CLASSES
#include "concretelang/Dialect/Concrete/Transforms/Passes.h.inc"

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<ModuleOp>> createAddRuntimeContext();
} // namespace concretelang
} // namespace mlir

#endif // CONCRETELANG_DIALECT_CONCRETE_TRANSFORMS_PASSES_H_

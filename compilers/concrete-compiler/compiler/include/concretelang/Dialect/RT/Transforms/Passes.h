// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_RT_TRANSFORMS_PASSES_H
#define CONCRETELANG_DIALECT_RT_TRANSFORMS_PASSES_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/Pass.h"

#include "concretelang/Dialect/RT/IR/RTDialect.h"

#define GEN_PASS_CLASSES
#include "concretelang/Dialect/RT/Transforms/Passes.h.inc"

namespace mlir {
namespace concretelang {
std::unique_ptr<OperationPass<func::FuncOp>> createHoistAwaitFuturePass();
} // namespace concretelang
} // namespace mlir

#endif // CONCRETELANG_DIALECT_RT_TRANSFORMS_PASSES_H

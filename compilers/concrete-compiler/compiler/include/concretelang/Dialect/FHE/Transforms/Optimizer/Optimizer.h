// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_OPTIMIZER_TRANSFORMS_PASSES_H
#define CONCRETELANG_DIALECT_OPTIMIZER_TRANSFORMS_PASSES_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <concretelang/Dialect/Optimizer/IR/OptimizerDialect.h>
#include <concretelang/Dialect/Optimizer/IR/OptimizerOps.h>
#include <concretelang/Support/V0Parameters.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/FHE/Transforms/Optimizer/Optimizer.h.inc>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createOptimizerPartitionFrontierMaterializationPass(
    const optimizer::CircuitSolution &solverSolution);

} // namespace concretelang
} // namespace mlir

#endif

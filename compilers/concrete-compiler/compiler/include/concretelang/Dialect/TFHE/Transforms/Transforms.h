// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TFHE_OPTIMIZATION_PASS_H
#define CONCRETELANG_TFHE_OPTIMIZATION_PASS_H

#include "concrete-optimizer.hpp"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "mlir/Pass/Pass.h"

#define GEN_PASS_CLASSES
#include "concretelang/Dialect/TFHE/Transforms/Transforms.h.inc"

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<>> createTFHEOptimizationPass();
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>>
createTFHEOperationTransformationsPass();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
    createTFHECircuitSolutionParametrizationPass(
        std::optional<concrete_optimizer::dag::CircuitSolution>);
} // namespace concretelang
} // namespace mlir

#endif

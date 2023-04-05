// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TRANSFORMS_PASS_H
#define CONCRETELANG_TRANSFORMS_PASS_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Transforms/Passes.h.inc>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createCollapseParallelLoops();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createForLoopToParallel();
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createBatchingPass(int64_t maxBatchSize = std::numeric_limits<int64_t>::max());
std::unique_ptr<OperationPass<ModuleOp>> createSCFForallToSCFForPass();
} // namespace concretelang
} // namespace mlir

#endif

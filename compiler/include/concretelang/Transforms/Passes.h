// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_TRANSFORMS_PASS_H
#define CONCRETELANG_TRANSFORMS_PASS_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/SCF/SCF.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Transforms/Passes.h.inc>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>> createForLoopToParallel();
} // namespace concretelang
} // namespace mlir

#endif

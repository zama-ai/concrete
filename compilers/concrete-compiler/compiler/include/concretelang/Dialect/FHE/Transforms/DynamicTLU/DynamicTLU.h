// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_FHE_DYNAMIC_TLU_PASS_H
#define CONCRETELANG_FHE_DYNAMIC_TLU_PASS_H

#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES

#include <concretelang/Dialect/FHE/Transforms/DynamicTLU/DynamicTLU.h.inc>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<mlir::func::FuncOp>> createDynamicTLUPass();
} // namespace concretelang
} // namespace mlir

#endif

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_CONCRETE_OPTIMIZATION_PASS_H
#define CONCRETELANG_CONCRETE_OPTIMIZATION_PASS_H

#include <concretelang/Dialect/Concrete/IR/ConcreteDialect.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/Concrete/Transforms/Optimization.h.inc>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<>> createConcreteOptimizationPass();
} // namespace concretelang
} // namespace mlir

#endif

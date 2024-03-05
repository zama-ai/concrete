// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_FHE_BOOLEAN_PASS_H
#define CONCRETELANG_FHE_BOOLEAN_PASS_H

#include <concretelang/Dialect/FHE/IR/FHEDialect.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/FHE/Transforms/Boolean/Boolean.h.inc>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::OperationPass<>> createFHEBooleanTransformPass();

} // namespace concretelang
} // namespace mlir

#endif

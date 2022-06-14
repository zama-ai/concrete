// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_ONE_SHOT_BUFFERIZE_DPS_WRAPPER_PASS_H
#define CONCRETELANG_ONE_SHOT_BUFFERIZE_DPS_WRAPPER_PASS_H

#include <mlir/Dialect/Bufferization/IR/Bufferization.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Transforms/OneShotBufferizeDPSWrapper.h.inc>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOneShotBufferizeDPSWrapperPass();
} // namespace concretelang
} // namespace mlir

#endif

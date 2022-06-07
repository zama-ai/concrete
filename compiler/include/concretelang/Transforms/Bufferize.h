// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_BUFFERIZE_PASS_H
#define CONCRETELANG_BUFFERIZE_PASS_H

#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Transforms/Bufferize.h.inc>

namespace mlir {
namespace concretelang {
std::unique_ptr<mlir::FunctionPass> createFinalizingBufferizePass();
} // namespace concretelang
} // namespace mlir

#endif

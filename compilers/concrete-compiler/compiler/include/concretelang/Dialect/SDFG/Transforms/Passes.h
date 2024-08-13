// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_SDFG_TRANSFORMS_PASS_H
#define CONCRETELANG_SDFG_TRANSFORMS_PASS_H

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Linalg/IR/Linalg.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/Pass/Pass.h>

#define GEN_PASS_CLASSES
#include <concretelang/Dialect/SDFG/Transforms/Passes.h.inc>

namespace mlir {
namespace concretelang {

std::unique_ptr<mlir::Pass> createSDFGBufferOwnershipPass();

} // namespace concretelang
} // namespace mlir

#endif

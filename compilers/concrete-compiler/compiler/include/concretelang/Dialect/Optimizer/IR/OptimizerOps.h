// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_OPTIMIZER_IR_OPTIMIZEROPS_H
#define CONCRETELANG_DIALECT_OPTIMIZER_IR_OPTIMIZEROPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>

#define GET_OP_CLASSES
#include "concretelang/Dialect/Optimizer/IR/OptimizerOps.h.inc"

#endif

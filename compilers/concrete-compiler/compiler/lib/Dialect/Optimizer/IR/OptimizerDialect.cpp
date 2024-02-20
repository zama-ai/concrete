// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/Optimizer/IR/OptimizerDialect.h"
#include "concretelang/Dialect/Optimizer/IR/OptimizerOps.h"

#include "concretelang/Dialect/Optimizer/IR/OptimizerOpsDialect.cpp.inc"

using namespace mlir::concretelang::Optimizer;

void OptimizerDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/Optimizer/IR/OptimizerOps.cpp.inc"
      >();
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/Tracing/IR/TracingDialect.h"
#include "concretelang/Dialect/Tracing/IR/TracingOps.h"

#include "concretelang/Dialect/Tracing/IR/TracingOpsDialect.cpp.inc"

#include "concretelang/Support/Constants.h"

using namespace mlir::concretelang::Tracing;

void TracingDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/Tracing/IR/TracingOps.cpp.inc"
      >();
}

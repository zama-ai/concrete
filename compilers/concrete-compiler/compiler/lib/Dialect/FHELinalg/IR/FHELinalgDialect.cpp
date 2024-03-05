// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.h"
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgTypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOpsTypes.cpp.inc"

#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOpsDialect.cpp.inc"

using namespace mlir::concretelang::FHELinalg;

void FHELinalgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/FHELinalg/IR/FHELinalgOpsTypes.cpp.inc"
      >();
}

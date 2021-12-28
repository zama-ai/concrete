// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h"
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h"
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgOpsTypes.cpp.inc"

#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgOpsDialect.cpp.inc"

using namespace mlir::concretelang::HLFHELinalg;

void HLFHELinalgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgOpsTypes.cpp.inc"
      >();
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.h"
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgTypes.h"

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOpsTypes.cpp.inc"

#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOpsDialect.cpp.inc"

using namespace mlir::zamalang::HLFHELinalg;

void HLFHELinalgDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zamalang/Dialect/HLFHELinalg/IR/HLFHELinalgOpsTypes.cpp.inc"
      >();
}

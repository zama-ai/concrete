// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/BConcrete/IR/BConcreteDialect.h"
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/BConcrete/IR/BConcreteOpsTypes.cpp.inc"

#include "concretelang/Dialect/BConcrete/IR/BConcreteOpsDialect.cpp.inc"

using namespace mlir::concretelang::BConcrete;

void BConcreteDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/BConcrete/IR/BConcreteOpsTypes.cpp.inc"
      >();
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/IR/Builders.h"

#include "concretelang/Dialect/SDFG/IR/SDFGDialect.h"
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"

using namespace mlir::concretelang::SDFG;

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.cpp.inc"

void SDFGDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/SDFG/IR/SDFGOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "concretelang/Dialect/SDFG/IR/SDFGAttributes.cpp.inc"
      >();
}

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/SDFG/IR/SDFGAttributes.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/SDFG/IR/SDFGDialect.cpp.inc"

void StreamType::print(mlir::AsmPrinter &p) const {
  p << "<" << getElementType() << ">";
}

mlir::Type StreamType::parse(mlir::AsmParser &p) {
  if (p.parseLess())
    return mlir::Type();

  mlir::Type t;
  if (p.parseType(t))
    return mlir::Type();

  if (p.parseGreater())
    return mlir::Type();

  mlir::Location loc = p.getEncodedSourceLoc(p.getNameLoc());

  return getChecked(loc, loc.getContext(), t);
}

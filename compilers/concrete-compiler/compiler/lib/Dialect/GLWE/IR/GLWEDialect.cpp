// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/GLWE/IR/GLWEDialect.h"
#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Dialect/GLWE/IR/GLWETypes.h"
#include "mlir/IR/DialectImplementation.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

GLWEExprAttr getGlweConstantExprAttr(double value,
                                     ::mlir::MLIRContext *context) {
  GLWEExpr expr = getGlweConstantExpr(value, context);
  return GLWEExprAttr::get(context, expr);
}

} // namespace GLWE
} // namespace concretelang
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWETypes.cpp.inc"

#define GET_OP_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEOps.cpp.inc"

#include "concretelang/Dialect/GLWE/IR/GLWEDialect.cpp.inc"

#include "concretelang/Support/Constants.h"

using namespace mlir::concretelang::GLWE;

void GLWEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/GLWE/IR/GLWEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/GLWE/IR/GLWETypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.cpp.inc"
      >();
}

// GLWEExpr ///////////////////////////////////////

::mlir::Attribute
mlir::concretelang::GLWE::GLWEExprAttr::parse(::mlir::AsmParser &parser,
                                              ::mlir::Type odsType) {
  // parse '<'
  if (parser.parseLess())
    return {};

  GLWEExpr result = GLWEExpr::parse(parser);

  // parse '>'
  if (parser.parseGreater())
    return {};
  return GLWEExprAttr::get(parser.getContext(), result);
}

void mlir::concretelang::GLWE::GLWEExprAttr::print(
    ::mlir::AsmPrinter &printer) const {
  printer << "<";
  this->getExpr().print(printer);
  printer << ">";
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOpsDialect.cpp.inc"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/Concrete/IR/ConcreteOpsTypes.cpp.inc"

using namespace mlir::concretelang::Concrete;

void ConcreteDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/Concrete/IR/ConcreteOpsTypes.cpp.inc"
      >();
}

mlir::Type GlweCiphertextType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return Type();
  int polynomialSize = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(polynomialSize))
    return Type();
  if (parser.parseComma())
    return Type();
  int glweDimension = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(glweDimension))
    return Type();
  if (parser.parseComma())
    return Type();

  int p = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(p))
    return Type();
  if (parser.parseGreater())
    return Type();
  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());
  return getChecked(loc, loc.getContext(), polynomialSize, glweDimension, p);
}

void GlweCiphertextType::print(mlir::AsmPrinter &p) const {
  p << "<";
  if (getImpl()->polynomialSize == -1)
    p << "_";
  else
    p << getImpl()->polynomialSize;
  p << ",";
  if (getImpl()->glweDimension == -1)
    p << "_";
  else
    p << getImpl()->glweDimension;
  p << ",";
  if (getImpl()->p == -1)
    p << "_";
  else
    p << getImpl()->p;
  p << ">";
}

void LweCiphertextType::print(mlir::AsmPrinter &p) const {
  p << "<";

  if (getDimension() == -1)
    p << "_";
  else
    p << getDimension();

  p << ",";
  if (getP() == -1)
    p << "_";
  else
    p << getP();
  p << ">";
}

mlir::Type LweCiphertextType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return mlir::Type();
  int dimension = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(dimension))
    return mlir::Type();
  if (parser.parseComma())
    return mlir::Type();
  int p = -1;
  if (parser.parseOptionalKeyword("_") && parser.parseInteger(p))
    return mlir::Type();
  if (parser.parseGreater())
    return mlir::Type();

  mlir::Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  return getChecked(loc, loc.getContext(), dimension, p);
}

void CleartextType::print(mlir::AsmPrinter &p) const {
  p << "<";
  if (getP() == -1)
    p << "_";
  else
    p << getP();
  p << ">";
}

mlir::Type CleartextType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return mlir::Type();

  int p = -1;

  if (parser.parseOptionalKeyword("_") && parser.parseInteger(p))
    return mlir::Type();
  if (parser.parseGreater())
    return mlir::Type();

  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  return getChecked(loc, loc.getContext(), p);
}

void PlaintextType::print(mlir::AsmPrinter &p) const {
  p << "<";
  if (getP() == -1)
    p << "_";
  else
    p << getP();
  p << ">";
}

mlir::Type PlaintextType::parse(mlir::AsmParser &parser) {

  if (parser.parseLess())
    return mlir::Type();

  int p = -1;

  if (parser.parseOptionalKeyword("_") && parser.parseInteger(p))
    return mlir::Type();
  if (parser.parseGreater())
    return mlir::Type();

  Location loc = parser.getEncodedSourceLoc(parser.getNameLoc());

  return getChecked(loc, loc.getContext(), p);
}

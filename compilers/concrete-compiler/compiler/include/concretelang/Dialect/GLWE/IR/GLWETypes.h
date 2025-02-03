// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWETYPES_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWETYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#include <mlir/Dialect/Arith/IR/Arith.h>

#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.h"

namespace mlir {

template <>
struct FieldParser<::llvm::SmallVector<concretelang::GLWE::GLWEExprAttr>> {
  static FailureOr<::llvm::SmallVector<concretelang::GLWE::GLWEExprAttr>>
  parse(AsmParser &parser) {
    if (parser.parseLParen())
      return failure();
    ::llvm::SmallVector<concretelang::GLWE::GLWEExprAttr> elements;
    auto elementParser = [&]() {
      auto element =
          FieldParser<concretelang::GLWE::GLWEExprAttr>::parse(parser);
      if (failed(element))
        return failure();
      elements.push_back(*element);
      return success();
    };
    if (parser.parseCommaSeparatedList(elementParser))
      return failure();
    if (parser.parseRParen())
      return failure();

    return elements;
  }
};

template <>
void AsmPrinter::printStrippedAttrOrType<concretelang::GLWE::GLWEExprAttr>(
    ::llvm::ArrayRef<concretelang::GLWE::GLWEExprAttr> attrOrTypes) {
  getStream() << "(";
  llvm::interleaveComma(attrOrTypes, getStream(),
                        [this](concretelang::GLWE::GLWEExprAttr attrOrType) {
                          printStrippedAttrOrType(attrOrType);
                        });
  getStream() << ")";
}
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWETypes.h.inc"

#endif

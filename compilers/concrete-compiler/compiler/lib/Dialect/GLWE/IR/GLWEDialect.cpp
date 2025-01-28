// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/GLWE/IR/GLWEDialect.h"
#include "concretelang/Dialect/GLWE/IR/GLWEOps.h"
#include "concretelang/Dialect/GLWE/IR/GLWETypes.h"
#include "mlir/IR/DialectImplementation.h"
// #include "mlir/lib/AsmParser/AsmParserImpl.h"

// #include "concretelang/Dialect/GLWE/Interfaces/GLWEInterfaces.h"

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

// class MyParser : public mlir::detail::AsmParserImpl<mlir::DialectAsmParser> {

// }

// GlweExpr ///////////////////////////////////////
::mlir::Attribute
mlir::concretelang::GLWE::VariableAttr::parse(::mlir::AsmParser &parser,
                                              ::mlir::Type odsType) {

  // auto parserImpl = mlir::dyn_cast<mlir::detail::AsmParserImpl>(parser);
  // parser.

  if (parser.parseLess())
    return {};
  StringAttr symbol;
  if (parser.parseSymbolName(symbol)) {
    return {};
  }
  if (parser.parseEqual()) {
    return {};
  }
  llvm::SmallVector<double> domain;
  if (parser.parseLSquare()) {
    return {};
  }
  if (failed(parser.parseCommaSeparatedList([&]() {
        double value;
        if (parser.parseFloat(value))
          return failure();
        domain.push_back(value);
        return success();
      }))) {
    return {};
  }
  if (parser.parseRSquare()) {
    return {};
  }
  Variable expr;
  return VariableAttr::get(parser.getContext(), expr);
}

void mlir::concretelang::GLWE::VariableAttr::print(
    ::mlir::AsmPrinter &printer) const {
  printer << "<TODO";

  printer << ">";
}

// GlweExpr ///////////////////////////////////////

GlweExpr parseGlweExpr(GlweExpr lhs, ::mlir::AsmParser &parser) {
  if (!lhs) {
    if (succeeded(parser.parseOptionalLParen())) {
      lhs = parseGlweExpr({}, parser);
      if (lhs && parser.parseRParen().succeeded()) {
        return parseGlweExpr(lhs, parser);
      }
      return {};
    }
    mlir::StringAttr symbol;
    double constant;
    if (succeeded(parser.parseOptionalSymbolName(symbol))) {
      lhs = getGlweSymbolExpr(symbol.getValue(), parser.getContext());
    } else if (succeeded(parser.parseFloat(constant))) {
      // TODO: putain d'optional!
      lhs = getGlweConstantExpr(constant, parser.getContext());
    }
    if (lhs) {
      return parseGlweExpr(lhs, parser);
    }
    std::optional<GlweExprKind> unKind;
    if (succeeded(parser.parseOptionalKeyword("abs"))) {
      unKind.emplace(GlweExprKind::Abs);
    } else if (succeeded(parser.parseOptionalKeyword("floor"))) {
      unKind.emplace(GlweExprKind::Floor);
    } else if (succeeded(parser.parseOptionalKeyword("ceil"))) {
      unKind.emplace(GlweExprKind::Ceil);
    }
    if (unKind.has_value()) {
      if (failed(parser.parseLParen()))
        return {};
      lhs = parseGlweExpr({}, parser);
      if (failed(parser.parseRParen()))
        return {};
      return getGlweUnaryExpr(unKind.value(), lhs, parser.getContext());
    }
    std::optional<GlweExprKind> binKind;
    if (succeeded(parser.parseOptionalKeyword("max"))) {
      binKind.emplace(GlweExprKind::Max);
    } else if (succeeded(parser.parseOptionalKeyword("min"))) {
      binKind.emplace(GlweExprKind::Min);
    }
    if (failed(parser.parseLParen())) {
      return {};
    }
    lhs = parseGlweExpr({}, parser);
    if (failed(parser.parseComma())) {
      return {};
    }
    auto rhs = parseGlweExpr({}, parser);
    if (failed(parser.parseRParen())) {
      return {};
    }
    return getGlweBinaryExpr(binKind.value(), lhs, rhs, parser.getContext());
  }
  std::optional<GlweExprKind> binKind;
  if (succeeded(parser.parseOptionalPlus())) {
    binKind.emplace(GlweExprKind::Add);
  } else if (succeeded(parser.parseOptionalStar())) {
    binKind.emplace(GlweExprKind::Mul);
    if (succeeded(parser.parseOptionalStar())) {
      binKind.emplace(GlweExprKind::Pow);
    }
  } else if (succeeded(parser.parseOptionalKeyword("div"))) {
    binKind.emplace(GlweExprKind::Div);
  }
  if (binKind.has_value()) {
    auto rhs = parseGlweExpr({}, parser);
    return getGlweBinaryExpr(binKind.value(), lhs, rhs, parser.getContext());
  }
  return lhs;
}

::mlir::Attribute
mlir::concretelang::GLWE::GlweExprAttr::parse(::mlir::AsmParser &parser,
                                              ::mlir::Type odsType) {
  // parse '<'
  if (parser.parseLess())
    return {};

  GlweExpr result = parseGlweExpr({}, parser);

  // parse '>'
  if (parser.parseGreater())
    return {};
  return GlweExprAttr::get(parser.getContext(), result);
}

void mlir::concretelang::GLWE::GlweExprAttr::print(
    ::mlir::AsmPrinter &printer) const {
  printer << "<";
  this->getExpr().print(printer);
  printer << ">";
}

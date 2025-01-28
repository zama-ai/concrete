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
GlweExpr parseGlweExpr(GlweExpr lhs, ::mlir::AsmParser &parser);

std::optional<GlweExprKind> parseLowPrecedenceKind(::mlir::AsmParser &parser) {
  std::optional<GlweExprKind> binKind;
  if (succeeded(parser.parseOptionalPlus())) {
    binKind.emplace(GlweExprKind::Add);
  }
  return binKind;
}

std::optional<GlweExprKind> parseHighPrecedenceKind(::mlir::AsmParser &parser) {
  std::optional<GlweExprKind> binKind;
  if (succeeded(parser.parseOptionalStar())) {
    binKind.emplace(GlweExprKind::Mul);
    if (succeeded(parser.parseOptionalStar())) {
      binKind.emplace(GlweExprKind::Pow);
    }
  } else if (succeeded(parser.parseOptionalKeyword("div"))) {
    binKind.emplace(GlweExprKind::Div);
  }
  return binKind;
}

std::optional<GlweExprKind> parseBinaryKind(::mlir::AsmParser &parser) {
  auto binKind = parseLowPrecedenceKind(parser);
  if (!binKind.has_value()) {
    binKind = parseHighPrecedenceKind(parser);
  }
  return binKind;
}

GlweExpr parseGlweOperandExpr(GlweExpr lhs, mlir::AsmParser &parser) {
  // llvm::errs() << "parseGlweOperandExpr 0\n";
  // llvm::errs() << parser.getCurrentLocation().getPointer();
  // llvm::errs() << "\n\n";

  if (auto kind = parseBinaryKind(parser); kind.has_value()) {
    // llvm::errs() << "Binary kind " << (unsigned)kind.value();
    if (lhs) {
      parser.emitError(parser.getCurrentLocation(),
                       "missing right operand of binary operator");
    } else {
      parser.emitError(parser.getCurrentLocation(),
                       "missing left operand of binary operator");
    }
    return {};
  }
  // llvm::errs() << "parseGlweOperandExpr 1\n";
  // Parse parenthetical expression
  if (parser.parseOptionalLParen().succeeded()) {
    lhs = parseGlweExpr({}, parser);
    if (lhs && parser.parseRParen().succeeded()) {
      // llvm::errs() << "done paren\n\n";
      return lhs;
    }
    return {};
  }
  mlir::StringAttr symbol;
  double constant;
  // llvm::errs() << "parseGlweOperandExpr 2\n";
  // Parse leaf expression (symbol and constant)
  if (succeeded(parser.parseOptionalSymbolName(symbol))) {
    return getGlweSymbolExpr(symbol.getValue(), parser.getContext());
  } else if (succeeded(parser.parseFloat(constant))) {
    // TODO: putain d'optional!
    return getGlweConstantExpr(constant, parser.getContext());
  }
  // llvm::errs() << "parseGlweOperandExpr 3\n";
  // Parse unary expression
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
  // llvm::errs() << "parseGlweOperandExpr 4\n";
  return nullptr;
}

GlweExpr parseHighPrecedenceExpr(GlweExpr llhs, GlweExprKind llhsOp,
                                 mlir::AsmParser &parser) {
  GlweExpr lhs;
  // llvm::errs() << "parseHighPrecedenceExpr 0\n";
  // llvm::errs() << parser.getCurrentLocation().getPointer();
  // llvm::errs() << "\n\n";
  lhs = parseGlweOperandExpr(llhs, parser);
  // llvm::errs() << "parseHighPrecedenceExpr 0.1\n";

  if (!lhs) {
    // llvm::errs() << "parseHighPrecedenceExpr 0.2\n";
    return {};
  }
  // found lhs
  // llvm::errs() << "parseHighPrecedenceExpr 1\n";
  if (auto hOp = parseHighPrecedenceKind(parser); hOp.has_value()) {
    if (llhs) {
      // llvm::errs() << "parseHighPrecedenceExpr 2\n";
      // llhs `llhsOp` lhs `hOp` ...
      auto expr = getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
      return parseHighPrecedenceExpr(expr, hOp.value(), parser);
    }
    // llvm::errs() << "parseHighPrecedenceExpr 3\n";
    return parseHighPrecedenceExpr(lhs, hOp.value(), parser);
  }
  // llhs `llhsOp` lhs
  if (llhs) {
    return getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
  }

  return lhs;
}

GlweExpr parseGlweLowPrecedenceExpr(GlweExpr llhs, GlweExprKind llhsOp,
                                    ::mlir::AsmParser &parser) {
  // llvm::errs() << "parseGlweLowPrecedenceExpr 0\n";
  // llvm::errs() << parser.getCurrentLocation().getPointer();
  // llvm::errs() << "\n\n";

  GlweExpr lhs = parseGlweOperandExpr(llhs, parser);
  // llvm::errs() << "parseGlweLowPrecedenceExpr 0.1\n";
  if (!lhs) {
    // llvm::errs() << "parseGlweLowPrecedenceExpr 0.2\n";
    return {};
  }
  // llvm::errs() << "PARSE cdj\n";

  // found lhs
  // try low precedence op
  if (auto lOp = parseLowPrecedenceKind(parser); lOp.has_value()) {
    // llvm::errs() << "PARSE xxx\n";

    if (llhs) {
      // (llhs `llhsOp` lhs) `lOp` ...
      auto expr = getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
      return parseGlweLowPrecedenceExpr(expr, lOp.value(), parser);
    }
    // lhs `lOp` ...
    return parseGlweLowPrecedenceExpr(lhs, lOp.value(), parser);
  }
  // try high precedence op
  if (auto hOp = parseHighPrecedenceKind(parser); hOp.has_value()) {
    // llvm::errs() << "PARSE yyy\n";

    auto highRes = parseHighPrecedenceExpr(lhs, hOp.value(), parser);
    if (!highRes) {
      // llvm::errs() << "PARSE ddd\n";

      return {};
    }
    // llvm::errs() << "PARSE dddd\n";
    // llvm::errs() << parser.getCurrentLocation().getPointer();

    auto expr =
        llhs ? getGlweBinaryExpr(llhsOp, llhs, highRes, parser.getContext())
             : highRes;

    if (auto lOp = parseLowPrecedenceKind(parser); lOp.has_value()) {
      // llvm::errs() << "PARSE aaa\n";
      return parseGlweLowPrecedenceExpr(expr, lOp.value(), parser);
    }
    // llvm::errs() << "PARSE bbb\n";
    return expr;
  }
  if (llhs) {
    return getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
  }
  return lhs;
}

GlweExpr parseGlweExpr(GlweExpr lhs, ::mlir::AsmParser &parser) {
  return parseGlweLowPrecedenceExpr(nullptr, GlweExprKind::NoOp, parser);
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

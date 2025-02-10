#include "GLWEExprDetail.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

GLWEExprKind GLWEExpr::getKind() const { return expr->kind; }

void GLWEExpr::print(mlir::AsmPrinter &printer) const {
  switch (this->getKind()) {
  case GLWEExprKind::SymbolId:
    this->dyn_cast<GlweSymbolExpr>().print(printer);
    return;
  case GLWEExprKind::Constant:
    this->dyn_cast<GlweConstantExpr>().print(printer);
    return;
  default:
    if (auto binExpr = this->dyn_cast<GlweBinaryExpr>()) {
      binExpr.print(printer);
      return;
    } else if (auto unExpr = this->dyn_cast<GlweUnaryExpr>()) {
      unExpr.print(printer);
      return;
    }
    break;
  }
  llvm_unreachable("unknown GLWEExpr");
};

mlir::AsmPrinter &operator<<(mlir::AsmPrinter &p, const GLWEExpr &expr) {
  expr.print(p);
  return p;
}

MLIRContext *GLWEExpr::getContext() const { return expr->context; }

GLWEExpr GLWEExpr::operator+(GLWEExpr other) const {
  return getGlweBinaryExpr(GLWEExprKind::Add, *this, other, getContext());
}

// GlweSymbolExpr
void GlweSymbolExpr::print(mlir::AsmPrinter &printer) const {
  printer.printSymbolName(getSymbolName());
}

llvm::StringRef GlweSymbolExpr::getSymbolName() const {
  return static_cast<GlweSymbolExpr::ImplType *>(expr)->symbolName;
}

GLWEExpr getGlweSymbolExpr(llvm::StringRef symbolName, MLIRContext *context) {
  auto assignCtx = [context](detail::GlweSymbolExprStorage *storage) {
    storage->context = context;
  };
  // TODO: which uniquer should we use, where to register?
  StorageUniquer &uniquer = context->getAffineUniquer();
  uniquer.registerParametricStorageType<detail::GlweSymbolExprStorage>();

  return uniquer.get<detail::GlweSymbolExprStorage>(
      assignCtx, static_cast<unsigned>(GLWEExprKind::SymbolId), symbolName);
}

// GlweConstantExpr
void GlweConstantExpr::print(mlir::AsmPrinter &printer) const {
  printer.printFloat(llvm::APFloat(getValue()));
}

double GlweConstantExpr::getValue() const {
  return static_cast<GlweConstantExpr::ImplType *>(expr)->value;
}

GLWEExpr getGlweConstantExpr(double value, MLIRContext *context) {
  auto assignCtx = [context](detail::GlweConstantExprStorage *storage) {
    storage->context = context;
  };
  // TODO: which uniquer should we use, where to register?
  StorageUniquer &uniquer = context->getAffineUniquer();
  uniquer.registerParametricStorageType<detail::GlweConstantExprStorage>();

  return uniquer.get<detail::GlweConstantExprStorage>(
      assignCtx, static_cast<unsigned>(GLWEExprKind::Constant), value);
}

// GlweUnaryExpr
void GlweUnaryExpr::print(mlir::AsmPrinter &printer) const {
  switch (getKind()) {
  case GLWEExprKind::Neg:
    printer << "- " << getOperand();
    return;
  case GLWEExprKind::Abs:
    printer << "abs(" << getOperand() << ")";
    return;
  case GLWEExprKind::Floor:
    printer << "floor(" << getOperand() << ")";
    return;
  case GLWEExprKind::Ceil:
    printer << "ceil(" << getOperand() << ")";
    return;
  default:
    break;
  }
  llvm_unreachable("unknown GlweUnaryExpr");
}

GLWEExpr GlweUnaryExpr::getOperand() const {
  return static_cast<GlweUnaryExpr::ImplType *>(expr)->operand;
}

GLWEExpr getGlweUnaryExpr(GLWEExprKind kind, GLWEExpr operand,
                          MLIRContext *context) {
  auto assignCtx = [context](detail::GlweUnaryExprStorage *storage) {
    storage->context = context;
  };
  // TODO: which uniquer should we use, where to register?
  StorageUniquer &uniquer = context->getAffineUniquer();
  uniquer.registerParametricStorageType<detail::GlweUnaryExprStorage>();

  return uniquer.get<detail::GlweUnaryExprStorage>(
      assignCtx, static_cast<unsigned>(kind), operand);
}

// GlweBinaryExpr
void printWithOptionalParen(GLWEExpr expr, mlir::AsmPrinter &printer) {
  auto kind = expr.getKind();
  if (kind == GLWEExprKind::Add || kind == GLWEExprKind::Sub ||
      kind == GLWEExprKind::Mul || kind == GLWEExprKind::Pow) {
    printer << "(" << expr << ")";
    return;
  }
  expr.print(printer);
}

void GlweBinaryExpr::print(mlir::AsmPrinter &printer) const {
  switch (getKind()) {
  case GLWEExprKind::Max:
    printer << "max(" << getLHS() << ", " << getRHS() << ")";
    return;
  case GLWEExprKind::Min:
    printer << "min(" << getLHS() << ", " << getRHS() << ")";
    return;
  default:
    break;
  }
  printWithOptionalParen(getLHS(), printer);
  switch (getKind()) {
  case GLWEExprKind::Add:
    printer << " + ";
    break;
  case GLWEExprKind::Sub:
    printer << " - ";
    break;

  case GLWEExprKind::Mul:
    printer << " * ";
    break;
  case GLWEExprKind::Pow:
    printer << " ** ";
    break;
  case GLWEExprKind::Div:
    printer << " / ";
    break;
  default:
    llvm_unreachable("unknown GlweBinaryExpr");
    break;
  }
  printWithOptionalParen(getRHS(), printer);
}

GLWEExpr GlweBinaryExpr::getLHS() const {
  return static_cast<GlweBinaryExpr::ImplType *>(expr)->lhs;
}

GLWEExpr GlweBinaryExpr::getRHS() const {
  return static_cast<GlweBinaryExpr::ImplType *>(expr)->rhs;
}

GLWEExpr getGlweBinaryExpr(GLWEExprKind kind, GLWEExpr lhs, GLWEExpr rhs,
                           MLIRContext *context) {
  auto assignCtx = [context](detail::GlweBinaryExprStorage *storage) {
    storage->context = context;
  };
  // TODO: which uniquer should we use, where to register?
  StorageUniquer &uniquer = context->getAffineUniquer();
  uniquer.registerParametricStorageType<detail::GlweBinaryExprStorage>();

  return uniquer.get<detail::GlweBinaryExprStorage>(
      assignCtx, static_cast<unsigned>(kind), lhs, rhs);
}

// GLWEExpr parser
GLWEExpr parseGLWEExpr(GLWEExpr lhs, ::mlir::AsmParser &parser);

std::optional<GLWEExprKind> parseLowPrecedenceKind(::mlir::AsmParser &parser) {
  std::optional<GLWEExprKind> binKind;
  if (succeeded(parser.parseOptionalPlus())) {
    binKind.emplace(GLWEExprKind::Add);
  }
  if (succeeded(parser.parseOptionalMinus())) {
    binKind.emplace(GLWEExprKind::Sub);
  }
  return binKind;
}

std::optional<GLWEExprKind> parseHighPrecedenceKind(::mlir::AsmParser &parser) {
  std::optional<GLWEExprKind> binKind;
  if (succeeded(parser.parseOptionalStar())) {
    binKind.emplace(GLWEExprKind::Mul);
    if (succeeded(parser.parseOptionalStar())) {
      binKind.emplace(GLWEExprKind::Pow);
    }
  } else if (succeeded(parser.parseOptionalSlash())) {
    binKind.emplace(GLWEExprKind::Div);
  }
  return binKind;
}

std::optional<GLWEExprKind> parseBinaryKind(::mlir::AsmParser &parser) {
  auto binKind = parseLowPrecedenceKind(parser);
  if (!binKind.has_value()) {
    binKind = parseHighPrecedenceKind(parser);
  }
  return binKind;
}

GLWEExpr parseGlweOperandExpr(GLWEExpr lhs, mlir::AsmParser &parser) {
  if (auto kind = parseBinaryKind(parser); kind.has_value()) {
    if (lhs) {
      parser.emitError(parser.getCurrentLocation(),
                       "missing right operand of binary operator");
    } else {
      if (kind.value() == GLWEExprKind::Sub) {
        if ((lhs = parseGLWEExpr({}, parser))) {
          return getGlweUnaryExpr(GLWEExprKind::Neg, lhs, parser.getContext());
        }
      }
      parser.emitError(parser.getCurrentLocation(),
                       "missing left operand of binary operator");
    }
    return {};
  }
  // Parse parenthetical expression
  if (parser.parseOptionalLParen().succeeded()) {
    lhs = parseGLWEExpr({}, parser);
    if (lhs && parser.parseRParen().succeeded()) {
      return lhs;
    }
    return {};
  }
  mlir::StringAttr symbol;
  double constant;
  // Parse leaf expression (symbol and constant)
  if (succeeded(parser.parseOptionalSymbolName(symbol))) {
    return getGlweSymbolExpr(symbol.getValue(), parser.getContext());
  } else if (succeeded(parser.parseOptionalFloat(constant))) {
    return getGlweConstantExpr(constant, parser.getContext());
  }
  // Parse unary expression
  std::optional<GLWEExprKind> unKind;
  if (succeeded(parser.parseOptionalKeyword("abs"))) {
    unKind.emplace(GLWEExprKind::Abs);
  } else if (succeeded(parser.parseOptionalKeyword("floor"))) {
    unKind.emplace(GLWEExprKind::Floor);
  } else if (succeeded(parser.parseOptionalKeyword("ceil"))) {
    unKind.emplace(GLWEExprKind::Ceil);
  }
  if (unKind.has_value()) {
    if (failed(parser.parseLParen()))
      return {};
    lhs = parseGLWEExpr({}, parser);
    if (failed(parser.parseRParen()))
      return {};
    return getGlweUnaryExpr(unKind.value(), lhs, parser.getContext());
  }
  // Parse binary expression
  std::optional<GLWEExprKind> binKind;
  if (succeeded(parser.parseOptionalKeyword("max"))) {
    binKind.emplace(GLWEExprKind::Max);
  } else if (succeeded(parser.parseOptionalKeyword("min"))) {
    binKind.emplace(GLWEExprKind::Min);
  }
  if (binKind.has_value()) {
    if (failed(parser.parseLParen())) {
      return {};
    }
    lhs = parseGLWEExpr({}, parser);
    if (failed(parser.parseComma())) {
      return {};
    }
    auto rhs = parseGLWEExpr({}, parser);
    if (failed(parser.parseRParen())) {
      return {};
    }
    return getGlweBinaryExpr(binKind.value(), lhs, rhs, parser.getContext());
  }
  return nullptr;
}

GLWEExpr parseHighPrecedenceExpr(GLWEExpr llhs, GLWEExprKind llhsOp,
                                 mlir::AsmParser &parser) {
  GLWEExpr lhs;
  lhs = parseGlweOperandExpr(llhs, parser);

  if (!lhs) {
    return {};
  }
  // found lhs
  if (auto hOp = parseHighPrecedenceKind(parser); hOp.has_value()) {
    if (llhs) {
      // llhs `llhsOp` lhs `hOp` ...
      auto expr = getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
      return parseHighPrecedenceExpr(expr, hOp.value(), parser);
    }
    return parseHighPrecedenceExpr(lhs, hOp.value(), parser);
  }
  // llhs `llhsOp` lhs
  if (llhs) {
    return getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
  }
  return lhs;
}

GLWEExpr parseGlweLowPrecedenceExpr(GLWEExpr llhs, GLWEExprKind llhsOp,
                                    ::mlir::AsmParser &parser) {
  GLWEExpr lhs = parseGlweOperandExpr(llhs, parser);
  if (!lhs) {
    return {};
  }
  // found lhs
  // try low precedence op
  if (auto lOp = parseLowPrecedenceKind(parser); lOp.has_value()) {
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
    auto highRes = parseHighPrecedenceExpr(lhs, hOp.value(), parser);
    if (!highRes) {
      return {};
    }
    auto expr =
        llhs ? getGlweBinaryExpr(llhsOp, llhs, highRes, parser.getContext())
             : highRes;

    if (auto lOp = parseLowPrecedenceKind(parser); lOp.has_value()) {
      return parseGlweLowPrecedenceExpr(expr, lOp.value(), parser);
    }
    return expr;
  }
  if (llhs) {
    return getGlweBinaryExpr(llhsOp, llhs, lhs, parser.getContext());
  }
  return lhs;
}

GLWEExpr parseGLWEExpr(GLWEExpr lhs, ::mlir::AsmParser &parser) {
  return parseGlweLowPrecedenceExpr(nullptr, GLWEExprKind::NoOp, parser);
}

GLWEExpr GLWEExpr::parse(::mlir::AsmParser &parser) {
  return parseGLWEExpr({}, parser);
}

} // namespace GLWE
} // namespace concretelang
} // namespace mlir

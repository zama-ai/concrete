#include "GlweExprDetail.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

GlweExprKind GlweExpr::getKind() const { return expr->kind; }

void GlweExpr::print(mlir::AsmPrinter &printer) {
  switch (this->getKind()) {
  case GlweExprKind::SymbolId:
    this->dyn_cast<GlweSymbolExpr>().print(printer);
    return;
  case GlweExprKind::Constant:
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
  llvm_unreachable("unknown GlweExpr");
};

// GlweSymbolExpr
void GlweSymbolExpr::print(mlir::AsmPrinter &printer) {
  printer.printSymbolName(getSymbolName());
}

llvm::StringRef GlweSymbolExpr::getSymbolName() const {
  return static_cast<GlweSymbolExpr::ImplType *>(expr)->symbolName;
}

GlweExpr getGlweSymbolExpr(llvm::StringRef symbolName, MLIRContext *context) {
  auto assignCtx = [context](detail::GlweSymbolExprStorage *storage) {
    storage->context = context;
  };
  // TODO: which uniquer should we use, where to register?
  StorageUniquer &uniquer = context->getAffineUniquer();
  uniquer.registerParametricStorageType<detail::GlweSymbolExprStorage>();

  return uniquer.get<detail::GlweSymbolExprStorage>(
      assignCtx, static_cast<unsigned>(GlweExprKind::SymbolId), symbolName);
}

// GlweConstantExpr
void GlweConstantExpr::print(mlir::AsmPrinter &printer) {
  printer.printFloat(llvm::APFloat(getValue()));
}

double GlweConstantExpr::getValue() const {
  return static_cast<GlweConstantExpr::ImplType *>(expr)->value;
}

GlweExpr getGlweConstantExpr(double value, MLIRContext *context) {
  auto assignCtx = [context](detail::GlweConstantExprStorage *storage) {
    storage->context = context;
  };
  // TODO: which uniquer should we use, where to register?
  StorageUniquer &uniquer = context->getAffineUniquer();
  uniquer.registerParametricStorageType<detail::GlweConstantExprStorage>();

  return uniquer.get<detail::GlweConstantExprStorage>(
      assignCtx, static_cast<unsigned>(GlweExprKind::Constant), value);
}

// GlweUnaryExpr
void GlweUnaryExpr::print(mlir::AsmPrinter &printer) {
  switch (getKind()) {
  case GlweExprKind::Neg:
    printer << "- ";
    getOperand().print(printer);
    return;
  case GlweExprKind::Abs:
    printer << "abs(";
    getOperand().print(printer);
    printer << ")";
    return;
  case GlweExprKind::Floor:
    printer << "floor(";
    getOperand().print(printer);
    printer << ")";
    return;
  case GlweExprKind::Ceil:
    printer << "ceil(";
    getOperand().print(printer);
    printer << ")";
    return;
  default:
    break;
  }
  llvm_unreachable("unknown GlweUnaryExpr");
}

GlweExpr GlweUnaryExpr::getOperand() const {
  return static_cast<GlweUnaryExpr::ImplType *>(expr)->operand;
}

GlweExpr getGlweUnaryExpr(GlweExprKind kind, GlweExpr operand,
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
void printWithOptionalParen(GlweExpr expr, mlir::AsmPrinter &printer) {
  auto kind = expr.getKind();
  if (kind == GlweExprKind::Add || kind == GlweExprKind::Sub ||
      kind == GlweExprKind::Mul || kind == GlweExprKind::Pow) {
    printer << "(";
    expr.print(printer);
    printer << ")";
    return;
  }
  expr.print(printer);
}

void GlweBinaryExpr::print(mlir::AsmPrinter &printer) {
  switch (getKind()) {
  case GlweExprKind::Max:
    printer << "max(";
    getLHS().print(printer);
    printer << ", ";
    getRHS().print(printer);
    printer << ")";
    return;
  case GlweExprKind::Min:
    printer << "min(";
    getLHS().print(printer);
    printer << ", ";
    getRHS().print(printer);
    printer << ")";
    return;
  default:
    break;
  }
  printWithOptionalParen(getLHS(), printer);
  switch (getKind()) {
  case GlweExprKind::Add:
    printer << " + ";
    break;
  case GlweExprKind::Sub:
    printer << " - ";
    break;

  case GlweExprKind::Mul:
    printer << " * ";
    break;
  case GlweExprKind::Pow:
    printer << " ** ";
    break;
  case GlweExprKind::Div:
    printer << " div ";
    break;
  default:
    llvm_unreachable("unknown GlweBinaryExpr");
    break;
  }
  printWithOptionalParen(getRHS(), printer);
}

GlweExpr GlweBinaryExpr::getLHS() const {
  return static_cast<GlweBinaryExpr::ImplType *>(expr)->lhs;
}

GlweExpr GlweBinaryExpr::getRHS() const {
  return static_cast<GlweBinaryExpr::ImplType *>(expr)->rhs;
}

GlweExpr getGlweBinaryExpr(GlweExprKind kind, GlweExpr lhs, GlweExpr rhs,
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

// GlweExpr parser
GlweExpr parseGlweExpr(GlweExpr lhs, ::mlir::AsmParser &parser);

std::optional<GlweExprKind> parseLowPrecedenceKind(::mlir::AsmParser &parser) {
  std::optional<GlweExprKind> binKind;
  if (succeeded(parser.parseOptionalPlus())) {
    binKind.emplace(GlweExprKind::Add);
  }
  if (succeeded(parser.parseOptionalMinus())) {
    binKind.emplace(GlweExprKind::Sub);
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
  if (auto kind = parseBinaryKind(parser); kind.has_value()) {
    if (lhs) {
      parser.emitError(parser.getCurrentLocation(),
                       "missing right operand of binary operator");
    } else {
      if (kind.value() == GlweExprKind::Sub) {
        if ((lhs = parseGlweExpr({}, parser))) {
          return getGlweUnaryExpr(GlweExprKind::Neg, lhs, parser.getContext());
        }
      }
      parser.emitError(parser.getCurrentLocation(),
                       "missing left operand of binary operator");
    }
    return {};
  }
  // Parse parenthetical expression
  if (parser.parseOptionalLParen().succeeded()) {
    lhs = parseGlweExpr({}, parser);
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
  } else if (succeeded(parser.parseFloat(constant))) {
    // TODO: putain d'optional!
    return getGlweConstantExpr(constant, parser.getContext());
  }
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
  // Parse binary expression
  std::optional<GlweExprKind> binKind;
  if (succeeded(parser.parseOptionalKeyword("max"))) {
    binKind.emplace(GlweExprKind::Max);
  } else if (succeeded(parser.parseOptionalKeyword("min"))) {
    binKind.emplace(GlweExprKind::Min);
  }
  if (binKind.has_value()) {
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
  return nullptr;
}

GlweExpr parseHighPrecedenceExpr(GlweExpr llhs, GlweExprKind llhsOp,
                                 mlir::AsmParser &parser) {
  GlweExpr lhs;
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

GlweExpr parseGlweLowPrecedenceExpr(GlweExpr llhs, GlweExprKind llhsOp,
                                    ::mlir::AsmParser &parser) {
  GlweExpr lhs = parseGlweOperandExpr(llhs, parser);
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

GlweExpr parseGlweExpr(GlweExpr lhs, ::mlir::AsmParser &parser) {
  return parseGlweLowPrecedenceExpr(nullptr, GlweExprKind::NoOp, parser);
}

GlweExpr GlweExpr::parse(::mlir::AsmParser &parser) {
  return parseGlweExpr({}, parser);
}

} // namespace GLWE
} // namespace concretelang
} // namespace mlir
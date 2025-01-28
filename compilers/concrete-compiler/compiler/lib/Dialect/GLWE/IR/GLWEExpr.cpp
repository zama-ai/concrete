#include "GlweExprDetail.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

GlweExprKind GlweExpr::getKind() const { return expr->kind; }

void GlweExpr::print(mlir::AsmPrinter &printer) {
  switch (this->getKind()) {
  case GlweExprKind::SymbolId:
    this->dyn_cast<GlweSymbolExpr>().print(printer);
    break;
  case GlweExprKind::Constant:
    this->dyn_cast<GlweConstantExpr>().print(printer);
    break;
  default:
    if (auto binExpr = this->dyn_cast<GlweBinaryExpr>()) {
      binExpr.print(printer);
      return;
    } else if (auto unExpr = this->dyn_cast<GlweUnaryExpr>()) {
      unExpr.print(printer);
      return;
    }
    assert(false && "NYI");
    break;
  }
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
  llvm::APFloat ap(getValue());
  printer.printFloat(ap);
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
  if (kind == GlweExprKind::Add || kind == GlweExprKind::Mul ||
      kind == GlweExprKind::Pow) {
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
    assert(false);
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

} // namespace GLWE
} // namespace concretelang
} // namespace mlir
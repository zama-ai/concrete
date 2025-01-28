
#include <mlir/IR/DialectImplementation.h>

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWEEXPR_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWEEXPR_H

namespace mlir {
namespace concretelang {
namespace GLWE {

namespace detail {
class GlweExprStorage;
class GlweUnaryExprStorage;
class GlweBinaryExprStorage;
class GlweSymbolExprStorage;
class GlweConstantExprStorage;
} // namespace detail

enum class GlweExprKind {
  NoOp,
  /// @brief Addition of two expressions, glwe-expr `+` glwe-expr.
  Add,
  /// @brief Substraction of two expressions, glwe-expr `-` glwe-expr.
  Sub,
  /// @brief Multiplication of two expressions, glwe-expr `*` glwe-expr.
  Mul,
  /// @brief Power of a base expression with an exponent expression, glwe-expr
  /// `**` glwe-expr.
  Pow,
  /// @brief Division of a dividend expression by a divisor expression,
  /// glwe-expr `/` glwe-expr.
  Div,
  /// @brief Maximum of two expressions, `max(` glwe-expr `,` glwe-expr `)`.
  Max,
  /// @brief Minimun of two expressions, `min(` glwe-expr `,` glwe-expr `)`r.
  Min,

  LAST_BINARY_OP = Min,

  /// @brief Negation of an expression, `-` glwe-expr.
  Neg,
  /// @brief Absolute value of an expression, `abs(` glwe-expr `)`.
  Abs,
  /// @brief Floor of an expression, `floor(` glwe-expr `)`.
  Floor,
  /// @brief Ceil of an expression, `ceil(` glwe-expr `)`.
  Ceil,

  LAST_UNARY_OP = Ceil,

  /// @brief Constant float.
  Constant,
  /// @brief Symbolic identifier.
  SymbolId,
};

class GlweExpr {
public:
  using ImplType = detail::GlweExprStorage;

  constexpr GlweExpr() {}
  /* implicit */ GlweExpr(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  bool operator==(GlweExpr other) const { return expr == other.expr; }
  bool operator!=(GlweExpr other) const { return !(*this == other); }

  explicit operator bool() const { return expr; }
  bool operator!() const { return expr == nullptr; }

  friend ::llvm::hash_code hash_value(GlweExpr arg);

  // Return the kind of the expression.
  GlweExprKind getKind() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer);

  template <typename U> constexpr bool isa() const;

  template <typename U> U dyn_cast() const;

  friend ::llvm::hash_code hash_value(AffineExpr arg);

protected:
  ImplType *expr{nullptr};
};

inline ::llvm::hash_code hash_value(GlweExpr arg) {
  return ::llvm::hash_value(arg.expr);
}

/// A symbolic identifier appearing in an affine expression.
class GlweSymbolExpr : public GlweExpr {
public:
  using ImplType = detail::GlweSymbolExprStorage;
  /* implicit */ GlweSymbolExpr(GlweExpr::ImplType *ptr) : GlweExpr(ptr) {};

  llvm::StringRef getSymbolName() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer);
};

/// An integer constant appearing in affine expression.
class GlweConstantExpr : public GlweExpr {
public:
  using ImplType = detail::GlweConstantExprStorage;
  /* implicit */ GlweConstantExpr(GlweExpr::ImplType *ptr = nullptr)
      : GlweExpr(ptr) {};

  double getValue() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer);
};

/// @brief A Glwe unary expression.
class GlweUnaryExpr : public GlweExpr {
public:
  using ImplType = detail::GlweUnaryExprStorage;
  /* implicit */ GlweUnaryExpr(GlweExpr::ImplType *ptr = nullptr)
      : GlweExpr(ptr) {};
  GlweExpr getOperand() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer);
};

/// @brief A Glwe binary expression.
class GlweBinaryExpr : public GlweExpr {
public:
  using ImplType = detail::GlweBinaryExprStorage;
  /* implicit */ GlweBinaryExpr(GlweExpr::ImplType *ptr = nullptr)
      : GlweExpr(ptr) {};
  GlweExpr getLHS() const;
  GlweExpr getRHS() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer);
};

template <typename U> constexpr bool GlweExpr::isa() const {
  if constexpr (std::is_same_v<U, GlweSymbolExpr>)
    return getKind() == GlweExprKind::SymbolId;
  if constexpr (std::is_same_v<U, GlweConstantExpr>)
    return getKind() == GlweExprKind::Constant;
  if constexpr (std::is_same_v<U, GlweBinaryExpr>)
    return getKind() <= GlweExprKind::LAST_BINARY_OP;
  if constexpr (std::is_same_v<U, GlweUnaryExpr>)
    return getKind() > GlweExprKind::LAST_BINARY_OP &&
           getKind() <= GlweExprKind::LAST_UNARY_OP;
  return false;
}

template <typename U> U GlweExpr::dyn_cast() const {
  if (isa<U>())
    return U(expr);
  return U(nullptr);
}

GlweExpr getGlweUnaryExpr(GlweExprKind kind, GlweExpr operand,
                          MLIRContext *context);
GlweExpr getGlweBinaryExpr(GlweExprKind kind, GlweExpr lhs, GlweExpr rhs,
                           MLIRContext *context);
GlweExpr getGlweSymbolExpr(llvm::StringRef symbolName, MLIRContext *context);
GlweExpr getGlweConstantExpr(double value, MLIRContext *context);

} // namespace GLWE
} // namespace concretelang
} // namespace mlir

namespace llvm {

// GlweExpr hash just like pointers
template <> struct DenseMapInfo<mlir::concretelang::GLWE::GlweExpr> {
  static mlir::concretelang::GLWE::GlweExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::concretelang::GLWE::GlweExpr(
        static_cast<mlir::concretelang::GLWE::GlweExpr::ImplType *>(pointer));
  }
  static mlir::concretelang::GLWE::GlweExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::concretelang::GLWE::GlweExpr(
        static_cast<mlir::concretelang::GLWE::GlweExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::concretelang::GLWE::GlweExpr val) {
    return mlir::concretelang::GLWE::hash_value(val);
  }
  static bool isEqual(mlir::concretelang::GLWE::GlweExpr LHS,
                      mlir::concretelang::GLWE::GlweExpr RHS) {
    return LHS == RHS;
  }
};

} // namespace llvm

#endif
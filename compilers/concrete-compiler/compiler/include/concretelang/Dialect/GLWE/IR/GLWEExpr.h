
#include <mlir/IR/DialectImplementation.h>

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWEExpr_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWEExpr_H

namespace mlir {
namespace concretelang {
namespace GLWE {

namespace detail {
class GLWEExprStorage;
class GlweUnaryExprStorage;
class GlweBinaryExprStorage;
class GlweSymbolExprStorage;
class GlweConstantExprStorage;
} // namespace detail

enum class GLWEExprKind {
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
  /// @brief Log of an expression to the base 2, `log2(` glwe-expr `)`.
  Log2,

  LAST_UNARY_OP = Log2,

  /// @brief Constant float.
  Constant,
  /// @brief Symbolic identifier.
  SymbolId,
};

class GLWEExpr {
public:
  using ImplType = detail::GLWEExprStorage;

  constexpr GLWEExpr() {}
  /* implicit */ GLWEExpr(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  bool operator==(const GLWEExpr &other) const { return expr == other.expr; }
  bool operator!=(const GLWEExpr &other) const { return !(*this == other); }

  explicit operator bool() const { return expr; }
  bool operator!() const { return expr == nullptr; }

  friend ::llvm::hash_code hash_value(const GLWEExpr &arg);

  // Return the kind of the expression.
  GLWEExprKind getKind() const;

  // Parse an expression.
  static GLWEExpr parse(mlir::AsmParser &parser);

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer) const;

  // Returns a simplified expression
  // Note: just constant folding for now
  GLWEExpr simplify() const;

  GLWEExpr replace(std::function<GLWEExpr(GLWEExpr e)> f) const;

  template <typename U> constexpr bool isa() const;

  template <typename U> U dyn_cast() const;

  MLIRContext *getContext() const;

  GLWEExpr operator+(GLWEExpr other) const;
  GLWEExpr operator+(double other) const;
  GLWEExpr operator+(std::string other) const;
  GLWEExpr operator-(GLWEExpr other) const;
  GLWEExpr operator-(double other) const;
  GLWEExpr operator*(GLWEExpr other) const;
  GLWEExpr operator*(double other) const;
  GLWEExpr operator/(GLWEExpr other) const;
  GLWEExpr operator/(double other) const;
  GLWEExpr pow(GLWEExpr other) const;
  GLWEExpr pow(double other) const;
  GLWEExpr max(GLWEExpr other) const;
  GLWEExpr max(double other) const;
  GLWEExpr min(GLWEExpr other) const;
  GLWEExpr min(double other) const;
  GLWEExpr abs() const;
  GLWEExpr floor() const;
  GLWEExpr ceil() const;
  GLWEExpr log2() const;

protected:
  ImplType *expr{nullptr};
};

inline ::llvm::hash_code hash_value(const GLWEExpr &arg) {
  return ::llvm::hash_value(arg.expr);
}

/// A symbolic identifier appearing in an affine expression.
class GlweSymbolExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweSymbolExprStorage;
  /* implicit */ GlweSymbolExpr(GLWEExpr::ImplType *ptr) : GLWEExpr(ptr) {};

  llvm::StringRef getSymbolName() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

/// An integer constant appearing in affine expression.
class GlweConstantExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweConstantExprStorage;
  /* implicit */ GlweConstantExpr(GLWEExpr::ImplType *ptr = nullptr)
      : GLWEExpr(ptr) {};

  double getValue() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

/// @brief A Glwe unary expression.
class GlweUnaryExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweUnaryExprStorage;
  /* implicit */ GlweUnaryExpr(GLWEExpr::ImplType *ptr = nullptr)
      : GLWEExpr(ptr) {};
  GLWEExpr getOperand() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

/// @brief A Glwe binary expression.
class GlweBinaryExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweBinaryExprStorage;
  /* implicit */ GlweBinaryExpr(GLWEExpr::ImplType *ptr = nullptr)
      : GLWEExpr(ptr) {};
  GLWEExpr getLHS() const;
  GLWEExpr getRHS() const;

  // Print the expression in the output stream.
  void print(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

template <typename U> constexpr bool GLWEExpr::isa() const {
  if constexpr (std::is_same_v<U, GlweSymbolExpr>)
    return getKind() == GLWEExprKind::SymbolId;
  if constexpr (std::is_same_v<U, GlweConstantExpr>)
    return getKind() == GLWEExprKind::Constant;
  if constexpr (std::is_same_v<U, GlweBinaryExpr>)
    return getKind() <= GLWEExprKind::LAST_BINARY_OP;
  if constexpr (std::is_same_v<U, GlweUnaryExpr>)
    return getKind() > GLWEExprKind::LAST_BINARY_OP &&
           getKind() <= GLWEExprKind::LAST_UNARY_OP;
  return false;
}

template <typename U> U GLWEExpr::dyn_cast() const {
  if (isa<U>())
    return U(expr);
  return U(nullptr);
}

GLWEExpr getGlweUnaryExpr(GLWEExprKind kind, GLWEExpr operand,
                          MLIRContext *context);
GLWEExpr getGlweBinaryExpr(GLWEExprKind kind, GLWEExpr lhs, GLWEExpr rhs,
                           MLIRContext *context);
GLWEExpr getGlweSymbolExpr(llvm::StringRef symbolName, MLIRContext *context);
GLWEExpr getGlweConstantExpr(double value, MLIRContext *context);

mlir::concretelang::GLWE::GLWEExpr
operator+(double lhs, mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr
operator-(double lhs, mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr
operator/(double lhs, mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr
operator*(double lhs, mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr pow(double lhs,
                                       mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr max(mlir::concretelang::GLWE::GLWEExpr lhs,
                                       mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr max(double lhs,
                                       mlir::concretelang::GLWE::GLWEExpr rhs);
mlir::concretelang::GLWE::GLWEExpr max(mlir::concretelang::GLWE::GLWEExpr lhs,
                                       double rhs);
} // namespace GLWE
} // namespace concretelang
} // namespace mlir

namespace llvm {

// GLWEExpr hash just like pointers
template <> struct DenseMapInfo<mlir::concretelang::GLWE::GLWEExpr> {
  static mlir::concretelang::GLWE::GLWEExpr getEmptyKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getEmptyKey();
    return mlir::concretelang::GLWE::GLWEExpr(
        static_cast<mlir::concretelang::GLWE::GLWEExpr::ImplType *>(pointer));
  }
  static mlir::concretelang::GLWE::GLWEExpr getTombstoneKey() {
    auto *pointer = llvm::DenseMapInfo<void *>::getTombstoneKey();
    return mlir::concretelang::GLWE::GLWEExpr(
        static_cast<mlir::concretelang::GLWE::GLWEExpr::ImplType *>(pointer));
  }
  static unsigned getHashValue(mlir::concretelang::GLWE::GLWEExpr val) {
    return mlir::concretelang::GLWE::hash_value(val);
  }
  static bool isEqual(mlir::concretelang::GLWE::GLWEExpr LHS,
                      mlir::concretelang::GLWE::GLWEExpr RHS) {
    return LHS == RHS;
  }
};

template <> struct DenseMapInfo<double> {
  static double getEmptyKey() { return 0.42; }
  static double getTombstoneKey() { return -1.23; }
  static unsigned getHashValue(double val) {
    return llvm::hash_value(llvm::APFloat(val));
  }
  static bool isEqual(double LHS, double RHS) { return LHS == RHS; }
};

} // namespace llvm

#endif

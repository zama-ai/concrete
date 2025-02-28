
#include <mlir/IR/DialectImplementation.h>

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWEExpr_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWEExpr_H

namespace mlir {
namespace concretelang {
namespace GLWE {
class GLWEExpr;

namespace detail {
class GLWEExprStorage;
class GlweUnaryExprStorage;
class GlweBinaryExprStorage;
class GlweSymbolExprStorage;
class GlweConstantExprStorage;

template <WalkOrder walkOrder = WalkOrder::PostOrder>
WalkResult walk(GLWEExpr expr, function_ref<WalkResult(GLWEExpr)> callback);
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

enum class GLWEComparisonOperator {
  Less,
  LessOrEqual,
  Equal,
  Greater,
  GreaterOrEqual
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
  void doPrint(mlir::AsmPrinter &printer) const;

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

  // Walks over the nodes of an AST of GLWE expressions rooted at
  // `this` similar to `Operation::walk()`. The type of the argument
  // of the callback function determines for which kind of nodes the
  // function is invoked. The return type may be `WalkResult`, in
  // which case the callback function can skip nodes or interrupt the
  // walk, or `void`, which causes the callback function to be invoked
  // for all nodes of the specified type.
  template <
      WalkOrder walkOrder = WalkOrder::PostOrder, typename FuncTy,
      typename ArgT = ::mlir::detail::first_argument<FuncTy>,
      typename RetT = decltype(std::declval<FuncTy>()(std::declval<ArgT>()))>
  RetT walk(FuncTy &&callback) {
    auto wrapperFn = [&](GLWEExpr expr) -> WalkResult {
      if (ArgT v = expr.dyn_cast<ArgT>()) {
        if constexpr (!std::is_same<RetT, void>::value) {
          return callback(v);
        } else {
          callback(v);
          return WalkResult::advance();
        }
      }

      return WalkResult::advance();
    };

    if constexpr (!std::is_same<RetT, void>::value) {
      return detail::walk<walkOrder>(*this, wrapperFn);
    } else {
      detail::walk<walkOrder>(*this, wrapperFn);
    }
  }

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
  /* implicit */ GlweSymbolExpr(GLWEExpr::ImplType *ptr) : GLWEExpr(ptr){};

  llvm::StringRef getSymbolName() const;

  // Print the expression in the output stream.
  void doPrint(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

/// An integer constant appearing in affine expression.
class GlweConstantExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweConstantExprStorage;
  /* implicit */ GlweConstantExpr(GLWEExpr::ImplType *ptr = nullptr)
      : GLWEExpr(ptr){};

  double getValue() const;

  // Print the expression in the output stream.
  void doPrint(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

/// @brief A Glwe unary expression.
class GlweUnaryExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweUnaryExprStorage;
  /* implicit */ GlweUnaryExpr(GLWEExpr::ImplType *ptr = nullptr)
      : GLWEExpr(ptr){};
  GLWEExpr getOperand() const;

  // Print the expression in the output stream.
  void doPrint(mlir::AsmPrinter &printer) const;

  GLWEExpr simplify() const;
};

/// @brief A Glwe binary expression.
class GlweBinaryExpr : public GLWEExpr {
public:
  using ImplType = detail::GlweBinaryExprStorage;
  /* implicit */ GlweBinaryExpr(GLWEExpr::ImplType *ptr = nullptr)
      : GLWEExpr(ptr){};
  GLWEExpr getLHS() const;
  GLWEExpr getRHS() const;

  // Print the expression in the output stream.
  void doPrint(mlir::AsmPrinter &printer) const;

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

namespace detail {
// Walks over the AST of GLWE expressions rooted at `expr` and invokes
// `callback̀ for each node encountered during the walk. If the
// callback function returns `WalkResult::interrupt`, the walk is
// interrupted immediately and no further invocation of callback takes
// place. If the callback function returns `WalkResult::skip()`, the
// walk omits all sub-expressions of `expr` not yet visited.
//
// The template parameter `walkOrder` defines whether the callback is
// invoked for a node before its children (`WalkOrder::PreOrder`) or
// after its children (`WalkOrder::PostOrder`).
template <WalkOrder walkOrder>
WalkResult walk(GLWEExpr expr, function_ref<WalkResult(GLWEExpr)> callback) {
  if constexpr (walkOrder == WalkOrder::PreOrder) {
    WalkResult preRes = callback(expr);

    if (preRes == WalkResult::interrupt()) {
      return WalkResult::interrupt();
    } else if (preRes == WalkResult::skip()) {
      return WalkResult::advance();
    }
  }

  if (GlweUnaryExpr ue = expr.dyn_cast<GlweUnaryExpr>()) {
    WalkResult opRes = walk<walkOrder>(ue.getOperand(), callback);

    if (opRes == WalkResult::interrupt()) {
      return WalkResult::interrupt();
    }
  } else if (GlweBinaryExpr be = expr.dyn_cast<GlweBinaryExpr>()) {
    WalkResult lhsRes = walk<walkOrder>(be.getLHS(), callback);

    if (lhsRes == WalkResult::interrupt())
      return WalkResult::interrupt();

    WalkResult rhsRes = walk<walkOrder>(be.getRHS(), callback);

    if (rhsRes == WalkResult::interrupt())
      return WalkResult::interrupt();
  }

  if constexpr (walkOrder == WalkOrder::PostOrder) {
    return callback(expr);
  }

  return WalkResult::advance();
}

} // namespace detail
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

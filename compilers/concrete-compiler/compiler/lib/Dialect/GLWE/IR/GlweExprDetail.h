#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWEEXPRDETAIL_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWEEXPRDETAIL_H

#include "concretelang/Dialect/GLWE/IR/GlweExpr.h"

namespace mlir {
class MLIRContext;
namespace concretelang {
namespace GLWE {

namespace detail {

struct GlweExprStorage : public StorageUniquer::BaseStorage {
  MLIRContext *context;
  GlweExprKind kind;
};

struct GlweSymbolExprStorage : public GlweExprStorage {
  using KeyTy = std::pair<unsigned, llvm::StringRef>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GlweExprKind>(key.first) &&
           symbolName == key.second;
  }

  static GlweSymbolExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweSymbolExprStorage>();
    result->kind = static_cast<GlweExprKind>(key.first);
    result->symbolName = key.second;
    return result;
  }

  /// Position of this identifier in the argument list.
  llvm::StringRef symbolName;
};

struct GlweConstantExprStorage : public GlweExprStorage {
  using KeyTy = std::pair<unsigned, unsigned>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GlweExprKind>(key.first) &&
           value == static_cast<double>(key.second);
  }

  static GlweConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweConstantExprStorage>();
    result->kind = static_cast<GlweExprKind>(key.first);
    result->value = static_cast<double>(key.second);
    return result;
  }

  /// Position of this identifier in the argument list.
  double value;
};

struct GlweUnaryExprStorage : public GlweExprStorage {
  using KeyTy = std::pair<unsigned, GlweExpr>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GlweExprKind>(key.first) &&
           operand == static_cast<GlweExpr>(key.second);
  }

  static GlweUnaryExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweUnaryExprStorage>();
    result->kind = static_cast<GlweExprKind>(key.first);
    result->operand = static_cast<GlweExpr>(key.second);
    return result;
  }

  GlweExpr operand;
};

struct GlweBinaryExprStorage : public GlweExprStorage {
  using KeyTy = std::tuple<unsigned, GlweExpr, GlweExpr>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GlweExprKind>(std::get<0>(key)) &&
           lhs == static_cast<GlweExpr>(std::get<1>(key)) &&
           rhs == static_cast<GlweExpr>(std::get<2>(key));
  }

  static GlweBinaryExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweBinaryExprStorage>();
    result->kind = static_cast<GlweExprKind>(std::get<0>(key));
    result->lhs = static_cast<GlweExpr>(std::get<1>(key));
    result->rhs = static_cast<GlweExpr>(std::get<2>(key));

    return result;
  }

  GlweExpr lhs;
  GlweExpr rhs;
};

} // namespace detail
} // namespace GLWE
} // namespace concretelang
} // namespace mlir

#endif
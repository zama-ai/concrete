#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWEExprDETAIL_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWEExprDETAIL_H

#include "concretelang/Dialect/GLWE/IR/GLWEExpr.h"

namespace mlir {
class MLIRContext;
namespace concretelang {
namespace GLWE {

namespace detail {

struct GLWEExprStorage : public StorageUniquer::BaseStorage {
  MLIRContext *context;
  GLWEExprKind kind;
};

struct GlweSymbolExprStorage : public GLWEExprStorage {
  using KeyTy = std::pair<unsigned, llvm::StringRef>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GLWEExprKind>(key.first) &&
           symbolName == key.second;
  }

  static GlweSymbolExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweSymbolExprStorage>();
    result->kind = static_cast<GLWEExprKind>(key.first);
    result->symbolName = key.second;
    return result;
  }

  llvm::StringRef symbolName;
};

struct GlweConstantExprStorage : public GLWEExprStorage {
  using KeyTy = std::pair<unsigned, double>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GLWEExprKind>(key.first) && value == key.second;
  }

  static GlweConstantExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweConstantExprStorage>();
    result->kind = static_cast<GLWEExprKind>(key.first);
    result->value = key.second;
    return result;
  }

  double value;
};

struct GlweUnaryExprStorage : public GLWEExprStorage {
  using KeyTy = std::pair<unsigned, GLWEExpr>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GLWEExprKind>(key.first) &&
           operand == static_cast<GLWEExpr>(key.second);
  }

  static GlweUnaryExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweUnaryExprStorage>();
    result->kind = static_cast<GLWEExprKind>(key.first);
    result->operand = static_cast<GLWEExpr>(key.second);
    return result;
  }

  GLWEExpr operand;
};

struct GlweBinaryExprStorage : public GLWEExprStorage {
  using KeyTy = std::tuple<unsigned, GLWEExpr, GLWEExpr>;

  bool operator==(const KeyTy &key) const {
    return kind == static_cast<GLWEExprKind>(std::get<0>(key)) &&
           lhs == static_cast<GLWEExpr>(std::get<1>(key)) &&
           rhs == static_cast<GLWEExpr>(std::get<2>(key));
  }

  static GlweBinaryExprStorage *
  construct(StorageUniquer::StorageAllocator &allocator, const KeyTy &key) {
    auto *result = allocator.allocate<GlweBinaryExprStorage>();
    result->kind = static_cast<GLWEExprKind>(std::get<0>(key));
    result->lhs = static_cast<GLWEExpr>(std::get<1>(key));
    result->rhs = static_cast<GLWEExpr>(std::get<2>(key));

    return result;
  }

  GLWEExpr lhs;
  GLWEExpr rhs;
};

} // namespace detail
} // namespace GLWE
} // namespace concretelang
} // namespace mlir

#endif
// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWEATTRS_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWEATTRS_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#include "concretelang/Dialect/GLWE/IR/GlweExpr.h"
#include "concretelang/Dialect/GLWE/IR/ParameterVariableDetail.h"

namespace mlir {
namespace concretelang {
namespace GLWE {

namespace detail {
class VariableStorage;
} // namespace detail

class Variable {
public:
  using ImplType = detail::VariableStorage;
  constexpr Variable() {}
  /* implicit */ Variable(const ImplType *expr)
      : expr(const_cast<ImplType *>(expr)) {}

  bool operator==(Variable other) const { return expr == other.expr; }
  bool operator!=(Variable other) const { return !(*this == other); }

  friend ::llvm::hash_code hash_value(Variable arg);

protected:
  ImplType *expr{nullptr};
};

inline ::llvm::hash_code hash_value(Variable arg) {
  return ::llvm::hash_value(arg.expr);
}

} // namespace GLWE
} // namespace concretelang

} // namespace mlir
#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.h.inc"

#endif

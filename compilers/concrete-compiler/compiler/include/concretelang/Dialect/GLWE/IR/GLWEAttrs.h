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

#include "concretelang/Dialect/GLWE/IR/GLWEExpr.h"

namespace mlir {
namespace concretelang {
namespace GLWE {
struct GLWESymbolName {
  GLWESymbolName(mlir::StringAttr name) : name(name) {}
  bool operator==(const GLWESymbolName &var) const {
    return var.getName() == getName();
  }

  mlir::StringAttr getName() const { return name; }
  mlir::StringAttr name;
};

::llvm::hash_code hash_value(const GLWESymbolName &arg) {
  return ::llvm::hash_value(arg.getName());
}

llvm::raw_ostream &
operator<<(llvm::raw_ostream &os,
           mlir::concretelang::GLWE::GLWESymbolName const &var) {
  return os << "@" << var.getName().strref();
}
} // namespace GLWE
} // namespace concretelang
} // namespace mlir

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.h.inc"

#endif

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

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.h.inc"

#endif

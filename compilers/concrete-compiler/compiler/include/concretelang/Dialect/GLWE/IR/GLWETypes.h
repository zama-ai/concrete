// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_GLWE_IR_GLWETYPES_H
#define CONCRETELANG_DIALECT_GLWE_IR_GLWETYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#include <mlir/Dialect/Arith/IR/Arith.h>

//#include "concretelang/Dialect/GLWE/Interfaces/GLWEInterfaces.h"
#include "concretelang/Dialect/GLWE/IR/GLWEAttrs.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/GLWE/IR/GLWETypes.h.inc"

#endif

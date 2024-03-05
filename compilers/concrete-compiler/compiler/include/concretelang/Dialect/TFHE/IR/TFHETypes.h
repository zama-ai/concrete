// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHETYPES_H
#define CONCRETELANG_DIALECT_TFHE_IR_TFHETYPES_H

#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"
#include "llvm/ADT/TypeSwitch.h"
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOpsTypes.h.inc"

#endif

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#ifndef ZAMALANG_DIALECT_BConcrete_BConcrete_OPS_H
#define ZAMALANG_DIALECT_BConcrete_BConcrete_OPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/BConcrete/IR/BConcreteOps.h.inc"

#endif

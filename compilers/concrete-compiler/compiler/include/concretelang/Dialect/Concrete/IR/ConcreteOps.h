// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_Concrete_Concrete_OPS_H
#define CONCRETELANG_DIALECT_Concrete_Concrete_OPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h.inc"

#endif

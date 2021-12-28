// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_DIALECT_MIDLFHE_IR_MIDLFHEOPS_H
#define CONCRETELANG_DIALECT_MIDLFHE_IR_MIDLFHEOPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "concretelang/Dialect/MidLFHE/IR/MidLFHETypes.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOps.h.inc"

#endif

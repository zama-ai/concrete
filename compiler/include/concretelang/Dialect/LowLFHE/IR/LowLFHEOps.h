// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#ifndef CONCRETELANG_DIALECT_LowLFHE_LowLFHE_OPS_H
#define CONCRETELANG_DIALECT_LowLFHE_LowLFHE_OPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "concretelang/Dialect/LowLFHE/IR/LowLFHETypes.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/LowLFHE/IR/LowLFHEOps.h.inc"

#endif

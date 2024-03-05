// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_RT_IR_RTOPS_H
#define CONCRETELANG_DIALECT_RT_IR_RTOPS_H

#include <mlir/Dialect/Bufferization/IR/AllocationOpInterface.h>
#include <mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/DataLayoutInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "concretelang/Dialect/RT/IR/RTTypes.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/RT/IR/RTOps.h.inc"

#endif

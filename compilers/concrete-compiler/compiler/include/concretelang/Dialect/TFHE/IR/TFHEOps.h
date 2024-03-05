// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_TFHE_IR_TFHEOPS_H
#define CONCRETELANG_DIALECT_TFHE_IR_TFHEOPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"
#include "concretelang/Interfaces/BatchableInterface.h"

#define GET_OP_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h.inc"

#endif

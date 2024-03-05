// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#ifndef CONCRETELANG_DIALECT_SDFG_IR_SDFGOPS_H
#define CONCRETELANG_DIALECT_SDFG_IR_SDFGOPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "concretelang/Dialect/SDFG/IR/SDFGEnums.h.inc"
#include "concretelang/Dialect/SDFG/IR/SDFGTypes.h"

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/SDFG/IR/SDFGAttributes.h.inc"

#define GET_OP_CLASSES
#include "concretelang/Dialect/SDFG/IR/SDFGOps.h.inc"

#endif

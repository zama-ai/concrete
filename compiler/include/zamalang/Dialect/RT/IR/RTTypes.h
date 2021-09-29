#ifndef ZAMALANG_DIALECT_RT_IR_RTTYPES_H
#define ZAMALANG_DIALECT_RT_IR_RTTYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/RT/IR/RTOpsTypes.h.inc"

#endif

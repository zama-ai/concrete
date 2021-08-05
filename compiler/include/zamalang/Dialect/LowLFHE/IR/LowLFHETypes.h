#ifndef ZAMALANG_DIALECT_LowLFHE_IR_LowLFHETYPES_H
#define ZAMALANG_DIALECT_LowLFHE_IR_LowLFHETYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOpsTypes.h.inc"

#endif

#ifndef ZAMALANG_DIALECT_MIDLFHE_IR_MIDLFHETYPES_H
#define ZAMALANG_DIALECT_MIDLFHE_IR_MIDLFHETYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.h.inc"

#endif

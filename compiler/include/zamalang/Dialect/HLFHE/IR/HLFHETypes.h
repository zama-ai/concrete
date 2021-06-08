#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHETYPES_H
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHETYPES_H

#include "llvm/ADT/TypeSwitch.h"
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/IR/DialectImplementation.h>

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOpsTypes.h.inc"

#endif

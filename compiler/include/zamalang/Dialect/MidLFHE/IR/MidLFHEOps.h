#ifndef ZAMALANG_DIALECT_MIDLFHE_IR_MIDLFHEOPS_H
#define ZAMALANG_DIALECT_MIDLFHE_IR_MIDLFHEOPS_H

#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h.inc"

#endif

#ifndef ZAMALANG_DIALECT_MidLFHE_MidLFHE_OPS_H
#define ZAMALANG_DIALECT_MidLFHE_MidLFHE_OPS_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>


#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h.inc"

#endif

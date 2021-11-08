#ifndef ZAMALANG_DIALECT_HLFHE_IR_HLFHEOPS_H
#define ZAMALANG_DIALECT_HLFHE_IR_HLFHEOPS_H

#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/BuiltinTypes.h>
#include <mlir/Interfaces/ControlFlowInterfaces.h>
#include <mlir/Interfaces/SideEffectInterfaces.h>

#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

namespace mlir {
namespace zamalang {
namespace HLFHE {

bool verifyEncryptedIntegerInputAndResultConsistency(
    OpState &op, EncryptedIntegerType &input, EncryptedIntegerType &result);

bool verifyEncryptedIntegerAndIntegerInputsConsistency(OpState &op,
                                                       EncryptedIntegerType &a,
                                                       IntegerType &b);

} // namespace HLFHE
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h.inc"

#endif

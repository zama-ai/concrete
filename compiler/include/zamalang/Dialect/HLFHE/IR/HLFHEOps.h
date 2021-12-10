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

/** Shared error message for all ApplyLookupTable variant Op (several Dialect)
 * E.g. HLFHE.apply_lookup_table(input, lut)
 * Message when the lut tensor has an invalid size,
 * i.e. it cannot accomodate the input elements bitwidth
 */
template <class Op>
void emitErrorBadLutSize(Op &op, std::string lutName, std::string inputName,
                         int expectedSize, int bitWidth) {
  auto s = op.emitOpError();
  s << ": `" << lutName << "` (operand #2)"
    << " inner dimension should have size " << expectedSize << "(=2^"
    << bitWidth << ") to match "
    << "`" << inputName << "` (operand #1)"
    << " elements bitwidth (" << bitWidth << ")";
}

} // namespace HLFHE
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h.inc"

#endif

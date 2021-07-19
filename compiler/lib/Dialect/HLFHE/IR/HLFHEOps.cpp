#include "mlir/IR/Region.h"

#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"
#include <mlir/IR/TypeUtilities.h>

namespace mlir {
namespace zamalang {
namespace HLFHE {

void Dot::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Side effects for Dot product: the first two operands are inputs,
  // the last one is an output
  effects.emplace_back(MemoryEffects::Read::get(), this->lhs(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Read::get(), this->rhs(),
                       SideEffects::DefaultResource::get());
  effects.emplace_back(MemoryEffects::Write::get(), this->out(),
                       SideEffects::DefaultResource::get());
}

} // namespace HLFHE
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.cpp.inc"

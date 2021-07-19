#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"

#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

namespace mlir {
namespace zamalang {
namespace HLFHE {

using mlir::zamalang::HLFHE::ApplyLookupTable;
using mlir::zamalang::HLFHE::EncryptedIntegerType;

::mlir::LogicalResult verifyApplyLookupTable(ApplyLookupTable &op) {
  auto ct = op.ct().getType().cast<EncryptedIntegerType>();
  auto l_cst = op.l_cst().getType().cast<MemRefType>();
  auto result = op.getResult().getType().cast<EncryptedIntegerType>();

  // Check the shape of l_cst argument
  auto width = ct.getWidth();
  auto lCstShape = l_cst.getShape();
  mlir::SmallVector<int64_t, 1> expectedShape{1 << width};
  if (!l_cst.hasStaticShape(expectedShape)) {
    op.emitOpError() << " should have as `l_cst` argument a shape of one "
                        "dimension equals to 2^p, where p is the width of the "
                        "`ct` argument.";
    return mlir::failure();
  }
  // Check the witdh of the encrypted integer and the integer of the tabulated
  // lambda are equals
  if (ct.getWidth() != l_cst.getElementType().cast<IntegerType>().getWidth()) {
    op.emitOpError()
        << " should have equals width beetwen the encrypted integer result and "
           "integers of the `tabulated_lambda` argument";
    return mlir::failure();
  }
  return mlir::success();
}

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

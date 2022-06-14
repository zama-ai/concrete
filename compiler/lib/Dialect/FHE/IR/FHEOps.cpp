// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

namespace mlir {
namespace concretelang {
namespace FHE {

bool verifyEncryptedIntegerInputAndResultConsistency(
    ::mlir::Operation &op, EncryptedIntegerType &input,
    EncryptedIntegerType &result) {
  if (input.getWidth() != result.getWidth()) {
    op.emitOpError(
        " should have the width of encrypted inputs and result equals");
    return false;
  }
  return true;
}

bool verifyEncryptedIntegerAndIntegerInputsConsistency(::mlir::Operation &op,
                                                       EncryptedIntegerType &a,
                                                       IntegerType &b) {
  if (a.getWidth() + 1 != b.getWidth()) {
    op.emitOpError(" should have the width of plain input equals to width of "
                   "encrypted input + 1");
    return false;
  }
  return true;
}

bool verifyEncryptedIntegerInputsConsistency(::mlir::Operation &op,
                                             EncryptedIntegerType &a,
                                             EncryptedIntegerType &b) {
  if (a.getWidth() != b.getWidth()) {
    op.emitOpError(" should have the width of encrypted inputs equals");
    return false;
  }
  return true;
}

::mlir::LogicalResult AddEintIntOp::verify() {
  auto a = this->a().getType().cast<EncryptedIntegerType>();
  auto b = this->b().getType().cast<IntegerType>();
  auto out = this->getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         a, b)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult AddEintOp::verify() {
  auto a = this->a().getType().cast<EncryptedIntegerType>();
  auto b = this->b().getType().cast<EncryptedIntegerType>();
  auto out = this->getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerInputsConsistency(*this->getOperation(), a, b)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult SubIntEintOp::verify() {
  auto a = this->a().getType().cast<IntegerType>();
  auto b = this->b().getType().cast<EncryptedIntegerType>();
  auto out = this->getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), b,
                                                       out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         b, a)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult NegEintOp::verify() {
  auto a = this->a().getType().cast<EncryptedIntegerType>();
  auto out = this->getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult MulEintIntOp::verify() {
  auto a = this->a().getType().cast<EncryptedIntegerType>();
  auto b = this->b().getType().cast<IntegerType>();
  auto out = this->getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(*this->getOperation(), a,
                                                       out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(*this->getOperation(),
                                                         a, b)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult ApplyLookupTableEintOp::verify() {
  auto ct = this->a().getType().cast<EncryptedIntegerType>();
  auto lut = this->lut().getType().cast<TensorType>();

  // Check the shape of lut argument
  auto width = ct.getWidth();
  auto expectedSize = 1 << width;

  mlir::SmallVector<int64_t, 1> expectedShape{expectedSize};
  if (!lut.hasStaticShape(expectedShape)) {
    emitErrorBadLutSize(*this, "lut", "ct", expectedSize, width);
    return mlir::failure();
  }
  if (!lut.getElementType().isInteger(64)) {
    this->emitOpError() << "should have the i64 constant";
    return mlir::failure();
  }
  return mlir::success();
}

// Avoid addition with constant 0
OpFoldResult AddEintIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2);
  auto toAdd = operands[1].dyn_cast_or_null<mlir::IntegerAttr>();
  if (toAdd != nullptr) {
    auto intToAdd = toAdd.getInt();
    if (intToAdd == 0) {
      return getOperand(0);
    }
  }
  return nullptr;
}

// Avoid multiplication with constant 1
OpFoldResult MulEintIntOp::fold(ArrayRef<Attribute> operands) {
  assert(operands.size() == 2);
  auto toMul = operands[1].dyn_cast_or_null<mlir::IntegerAttr>();
  if (toMul != nullptr) {
    auto intToMul = toMul.getInt();
    if (intToMul == 1) {
      return getOperand(0);
    }
  }
  return nullptr;
}

} // namespace FHE
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/FHE/IR/FHEOps.cpp.inc"

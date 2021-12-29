// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

namespace mlir {
namespace concretelang {
namespace FHE {

bool verifyEncryptedIntegerInputAndResultConsistency(
    ::mlir::OpState &op, EncryptedIntegerType &input,
    EncryptedIntegerType &result) {
  if (input.getWidth() != result.getWidth()) {
    op.emitOpError(
        " should have the width of encrypted inputs and result equals");
    return false;
  }
  return true;
}

bool verifyEncryptedIntegerAndIntegerInputsConsistency(::mlir::OpState &op,
                                                       EncryptedIntegerType &a,
                                                       IntegerType &b) {
  if (a.getWidth() + 1 != b.getWidth()) {
    op.emitOpError(" should have the width of plain input equals to width of "
                   "encrypted input + 1");
    return false;
  }
  return true;
}

bool verifyEncryptedIntegerInputsConsistency(::mlir::OpState &op,
                                             EncryptedIntegerType &a,
                                             EncryptedIntegerType &b) {
  if (a.getWidth() != b.getWidth()) {
    op.emitOpError(" should have the width of encrypted inputs equals");
    return false;
  }
  return true;
}

::mlir::LogicalResult verifyAddEintIntOp(AddEintIntOp &op) {
  auto a = op.a().getType().cast<EncryptedIntegerType>();
  auto b = op.b().getType().cast<IntegerType>();
  auto out = op.getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(op, a, out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(op, a, b)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult verifyAddEintOp(AddEintOp &op) {
  auto a = op.a().getType().cast<EncryptedIntegerType>();
  auto b = op.b().getType().cast<EncryptedIntegerType>();
  auto out = op.getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(op, a, out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerInputsConsistency(op, a, b)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult verifySubIntEintOp(SubIntEintOp &op) {
  auto a = op.a().getType().cast<IntegerType>();
  auto b = op.b().getType().cast<EncryptedIntegerType>();
  auto out = op.getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(op, b, out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(op, b, a)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult verifyNegEintOp(NegEintOp &op) {
  auto a = op.a().getType().cast<EncryptedIntegerType>();
  auto out = op.getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(op, a, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult verifyMulEintIntOp(MulEintIntOp &op) {
  auto a = op.a().getType().cast<EncryptedIntegerType>();
  auto b = op.b().getType().cast<IntegerType>();
  auto out = op.getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerInputAndResultConsistency(op, a, out)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(op, a, b)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

::mlir::LogicalResult verifyApplyLookupTable(ApplyLookupTableEintOp &op) {
  auto ct = op.ct().getType().cast<EncryptedIntegerType>();
  auto l_cst = op.l_cst().getType().cast<TensorType>();
  auto result = op.getResult().getType().cast<EncryptedIntegerType>();

  // Check the shape of l_cst argument
  auto width = ct.getWidth();
  auto expectedSize = 1 << width;
  auto lCstShape = l_cst.getShape();
  mlir::SmallVector<int64_t, 1> expectedShape{expectedSize};
  if (!l_cst.hasStaticShape(expectedShape)) {
    emitErrorBadLutSize(op, "l_cst", "ct", expectedSize, width);
    return mlir::failure();
  }
  if (!l_cst.getElementType().isInteger(64)) {
    op.emitOpError() << "should have the i64 constant";
    return mlir::failure();
  }
  return mlir::success();
}

} // namespace FHE
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/FHE/IR/FHEOps.cpp.inc"

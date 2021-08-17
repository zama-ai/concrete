#include "mlir/IR/Region.h"
#include "mlir/IR/TypeUtilities.h"

#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

namespace mlir {
namespace zamalang {
namespace HLFHE {

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

::mlir::LogicalResult verifyDotEintInt(Dot &op) {
  if (::mlir::failed(mlir::verifyCompatibleShape(op.lhs().getType(),
                                                 op.rhs().getType()))) {
    return op.emitOpError("arguments have incompatible shapes");
  }
  auto lhsEltType = op.lhs()
                        .getType()
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<EncryptedIntegerType>();
  auto rhsEltType = op.rhs()
                        .getType()
                        .cast<mlir::TensorType>()
                        .getElementType()
                        .cast<mlir::IntegerType>();
  auto resultType = op.getResult().getType().cast<EncryptedIntegerType>();
  if (!verifyEncryptedIntegerAndIntegerInputsConsistency(op, lhsEltType,
                                                         rhsEltType)) {
    return ::mlir::failure();
  }
  if (!verifyEncryptedIntegerInputAndResultConsistency(op, lhsEltType,
                                                       resultType)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

} // namespace HLFHE
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.cpp.inc"

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "mlir/IR/Region.h"

#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace mlir {
namespace concretelang {
namespace TFHE {

void emitOpErrorForKeyMismatch(mlir::OpState &op) {
  op.emitOpError() << "should have the same GLWE Secret Key";
}

mlir::LogicalResult _verifyGLWEIntegerOperator(mlir::OpState &op,
                                               GLWECipherTextType &a,
                                               IntegerType &b,
                                               GLWECipherTextType &result) {
  // verify consistency of a and result GLWE secret key
  if (a.getKey() != result.getKey()) {
    emitOpErrorForKeyMismatch(op);
    return mlir::failure();
  }

  // verify consistency of width of inputs
  if ((int)b.getWidth() != 64) {
    op.emitOpError() << "should have the width of `b` equals 64 : "
                     << b.getWidth() << " != 64";
    return mlir::failure();
  }
  return mlir::success();
}

template <class Operator>
mlir::LogicalResult verifyGLWEIntegerOperator(Operator &op) {
  auto a = ((mlir::Type)(op.getA().getType())).cast<GLWECipherTextType>();
  auto b = ((mlir::Type)(op.getB().getType())).cast<IntegerType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  return _verifyGLWEIntegerOperator(op, a, b, result);
}

template <class Operator>
mlir::LogicalResult verifyIntegerGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.getA().getType())).cast<IntegerType>();
  auto b = ((mlir::Type)(op.getB().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  return _verifyGLWEIntegerOperator(op, b, a, result);
}

/// verifyBinaryGLWEOperator verify parameters of operators that have the same
/// secret key.
template <class Operator>
mlir::LogicalResult verifyBinaryGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.getA().getType())).cast<GLWECipherTextType>();
  auto b = ((mlir::Type)(op.getB().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  // verify consistency of a and result GLWE secret key.
  if (a.getKey() != b.getKey() || a.getKey() != result.getKey()) {
    emitOpErrorForKeyMismatch(op);
    return mlir::failure();
  }

  return mlir::success();
}

/// verifyUnaryGLWEOperator verify parameters of operators that have the same
/// secret key.
template <class Operator>
mlir::LogicalResult verifyUnaryGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.getA().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  // verify consistency of a and result GLWE secret key.
  if (a.getKey() != result.getKey()) {
    emitOpErrorForKeyMismatch(op);
    return mlir::failure();
  }

  return mlir::success();
}

mlir::LogicalResult AddGLWEIntOp::verify() {
  return mlir::concretelang::TFHE::verifyGLWEIntegerOperator<AddGLWEIntOp>(
      *this);
}

mlir::LogicalResult AddGLWEOp::verify() {
  return ::mlir::concretelang::TFHE::verifyBinaryGLWEOperator<AddGLWEOp>(*this);
}

mlir::LogicalResult SubGLWEIntOp::verify() {
  return ::mlir::concretelang::TFHE::verifyIntegerGLWEOperator<SubGLWEIntOp>(
      *this);
}

mlir::LogicalResult NegGLWEOp::verify() {
  return ::mlir::concretelang::TFHE::verifyUnaryGLWEOperator<NegGLWEOp>(*this);
}

mlir::LogicalResult MulGLWEIntOp::verify() {
  return mlir::concretelang::TFHE::verifyGLWEIntegerOperator<MulGLWEIntOp>(
      *this);
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOps.cpp.inc"

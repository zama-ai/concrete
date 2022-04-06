// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "mlir/IR/Region.h"

#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace mlir {
namespace concretelang {
namespace TFHE {

void emitOpErrorForIncompatibleGLWEParameter(mlir::OpState &op,
                                             llvm::Twine parameter) {
  op.emitOpError() << "should have the same GLWE '" << parameter
                   << "' parameter";
}

mlir::LogicalResult _verifyGLWEIntegerOperator(mlir::OpState &op,
                                               GLWECipherTextType &a,
                                               IntegerType &b,
                                               GLWECipherTextType &result) {
  // verify consistency of a and result GLWE parameter
  if (a.getDimension() != result.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return mlir::failure();
  }
  if (a.getPolynomialSize() != result.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return mlir::failure();
  }
  if (a.getBits() != result.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return mlir::failure();
  }
  if (a.getP() != result.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return mlir::failure();
  }

  // verify consistency of width of inputs
  if ((int)b.getWidth() > a.getP() + 1) {
    op.emitOpError()
        << "should have the width of `b` equals or less than 'p'+1: "
        << b.getWidth() << " <= " << a.getP() << "+ 1";
    return mlir::failure();
  }
  return mlir::success();
}

// verifyGLWEIntegerOperator verify parameters of operators that has the
// following signature (!TFHE.glwe<{dim,poly,bits}{p}>, ip+1) ->
// (!TFHE.glwe<{dim,poly,bits}{p}>))
template <class Operator>
mlir::LogicalResult verifyGLWEIntegerOperator(Operator &op) {
  auto a = ((mlir::Type)(op.a().getType())).cast<GLWECipherTextType>();
  auto b = ((mlir::Type)(op.b().getType())).cast<IntegerType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  return _verifyGLWEIntegerOperator(op, a, b, result);
}

// verifyIntegerGLWEOperator verify parameters of operators that has the
// following signature (ip+1, !TFHE.glwe<{dim,poly,bits}{p}>) ->
// (!TFHE.glwe<{dim,poly,bits}{p}>))
template <class Operator>
mlir::LogicalResult verifyIntegerGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.a().getType())).cast<IntegerType>();
  auto b = ((mlir::Type)(op.b().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  return _verifyGLWEIntegerOperator(op, b, a, result);
}

// verifyBinaryGLWEOperator verify parameters of operators that has the
// following signature (!TFHE.glwe<{dim,poly,bits}{p}>,
// !TFHE.glwe<{dim,poly,bits}{p}>) ->
// (!TFHE.glwe<{dim,poly,bits}{p}>))
template <class Operator>
mlir::LogicalResult verifyBinaryGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.a().getType())).cast<GLWECipherTextType>();
  auto b = ((mlir::Type)(op.b().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  // verify consistency of a and result GLWE parameter
  if (a.getDimension() != b.getDimension() ||
      a.getDimension() != result.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return mlir::failure();
  }
  if (a.getPolynomialSize() != b.getPolynomialSize() ||
      a.getPolynomialSize() != result.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return mlir::failure();
  }
  if (a.getBits() != b.getBits() || a.getBits() != result.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return mlir::failure();
  }
  if (a.getP() != b.getP() || a.getP() != result.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return mlir::failure();
  }

  return mlir::success();
}

// verifyUnaryGLWEOperator verify parameters of operators that has the following
// signature (!TFHE.glwe<{dim,poly,bits}{p}>) ->
// (!TFHE.glwe<{dim,poly,bits}{p}>))
template <class Operator>
mlir::LogicalResult verifyUnaryGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.a().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  // verify consistency of a and result GLWE parameter
  if (a.getDimension() != result.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return mlir::failure();
  }
  if (a.getPolynomialSize() != result.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return mlir::failure();
  }
  if (a.getBits() != result.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return mlir::failure();
  }
  if (a.getP() != result.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOps.cpp.inc"

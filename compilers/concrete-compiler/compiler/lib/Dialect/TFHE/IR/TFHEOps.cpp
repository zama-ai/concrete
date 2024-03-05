// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/IR/Region.h"

#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.h"
#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

namespace mlir {
namespace concretelang {
namespace TFHE {

static const int64_t kUndefined = -1;

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

mlir::LogicalResult EncodeExpandLutForBootstrapOp::verify() {
  mlir::IntegerAttr polySizeAttr = this->getPolySizeAttr();

  mlir::RankedTensorType rtt =
      this->getResult().getType().template cast<mlir::RankedTensorType>();

  if (rtt.getNumElements() != polySizeAttr.getInt()) {
    this->emitError("The number of elements of the output tensor of ")
        << rtt.getNumElements()
        << " does not match the size of the polynomial of "
        << polySizeAttr.getInt();

    return mlir::failure();
  }

  return mlir::success();
}

template <typename BootstrapOpT>
mlir::LogicalResult verifyBootstrapSingleLUTConstraints(BootstrapOpT &op) {
  GLWEBootstrapKeyAttr keyAttr = op.getKeyAttr();

  if (keyAttr) {
    mlir::RankedTensorType rtt =
        op.getLookupTable().getType().template cast<mlir::RankedTensorType>();

    assert(rtt.getShape().size() == 1);

    // Do not fail on unparametrized ops
    if (keyAttr.getPolySize() == kUndefined)
      return mlir::success();

    if (rtt.getShape()[0] != keyAttr.getPolySize()) {
      op.emitError("Size of the lookup table of ")
          << rtt.getShape()[0] << " does not match the size of the polynom of "
          << keyAttr.getPolySize();

      return mlir::failure();
    }
  }

  return mlir::success();
}

mlir::LogicalResult BootstrapGLWEOp::verify() {
  return verifyBootstrapSingleLUTConstraints(*this);
}

mlir::LogicalResult BatchedBootstrapGLWEOp::verify() {
  return verifyBootstrapSingleLUTConstraints(*this);
}

mlir::LogicalResult BatchedMappedBootstrapGLWEOp::verify() {
  GLWEBootstrapKeyAttr keyAttr = this->getKeyAttr();

  if (keyAttr) {
    mlir::RankedTensorType lutRtt =
        this->getLookupTable().getType().cast<mlir::RankedTensorType>();

    mlir::RankedTensorType ciphertextsRtt =
        this->getCiphertexts().getType().cast<mlir::RankedTensorType>();

    assert(lutRtt.getShape().size() == 2);

    if (lutRtt.getShape()[1] != keyAttr.getPolySize()) {
      this->emitError("Size of the lookup table of ")
          << lutRtt.getShape()[1]
          << " does not match the size of the polynom of "
          << keyAttr.getPolySize();

      return mlir::failure();
    }

    if (lutRtt.getShape()[0] != ciphertextsRtt.getShape()[0]) {
      this->emitError("Number of lookup tables of ")
          << lutRtt.getShape()[0] << " does not match number of ciphertexts of "
          << ciphertextsRtt.getShape()[0];

      return mlir::failure();
    }
  }

  return mlir::success();
}

} // namespace TFHE
} // namespace concretelang
} // namespace mlir

#define GET_OP_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOps.cpp.inc"

#include "mlir/IR/Region.h"

#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace mlir {
namespace zamalang {
namespace MidLFHE {

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
  if (b.getWidth() > a.getP() + 1) {
    op.emitOpError()
        << "should have the width of `b` equals or less than 'p'+1: "
        << b.getWidth() << " <= " << a.getP() << "+ 1";
    return mlir::failure();
  }
  return mlir::success();
}

// verifyGLWEIntegerOperator verify parameters of operators that has the
// following signature (!MidLFHE.glwe<{dim,poly,bits}{p}>, ip+1) ->
// (!MidLFHE.glwe<{dim,poly,bits}{p}>))
template <class Operator>
mlir::LogicalResult verifyGLWEIntegerOperator(Operator &op) {
  auto a = ((mlir::Type)(op.a().getType())).cast<GLWECipherTextType>();
  auto b = ((mlir::Type)(op.b().getType())).cast<IntegerType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  return _verifyGLWEIntegerOperator(op, a, b, result);
}

// verifyIntegerGLWEOperator verify parameters of operators that has the
// following signature (ip+1, !MidLFHE.glwe<{dim,poly,bits}{p}>) ->
// (!MidLFHE.glwe<{dim,poly,bits}{p}>))
template <class Operator>
mlir::LogicalResult verifyIntegerGLWEOperator(Operator &op) {
  auto a = ((mlir::Type)(op.a().getType())).cast<IntegerType>();
  auto b = ((mlir::Type)(op.b().getType())).cast<GLWECipherTextType>();
  auto result =
      ((mlir::Type)(op.getResult().getType())).cast<GLWECipherTextType>();

  return _verifyGLWEIntegerOperator(op, b, a, result);
}

// verifyIntegerOperator verify parameters of operators that has the following
// signature (!MidLFHE.glwe<{dim,poly,bits}{p}>,
// !MidLFHE.glwe<{dim,poly,bits}{p}>) ->
// (!MidLFHE.glwe<{dim,poly,bits}{p}>))
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

/// verifyApplyLookupTable verify the GLWE parameters follow the rules:
/// - The l_cst argument must be a memref of one dimension of size 2^p
/// - The lookup table contains integer values of the same width of the output
mlir::LogicalResult verifyApplyLookupTable(ApplyLookupTable &op) {
  auto ct = op.ct().getType().cast<GLWECipherTextType>();
  auto l_cst = op.l_cst().getType().cast<RankedTensorType>();
  auto result = op.getResult().getType().cast<GLWECipherTextType>();

  // Check the shape of l_cst argument
  auto width = ct.getP();
  auto lCstShape = l_cst.getShape();
  mlir::SmallVector<int64_t, 1> expectedShape{1 << width};
  if (!l_cst.hasStaticShape(expectedShape)) {
    op.emitOpError() << "should have as `l_cst` argument a shape of one "
                        "dimension equals to 2^p, where p is the width of the "
                        "`ct` argument.";
    return mlir::failure();
  }
  // Disable this check for the moment
  // Check the witdh of the encrypted integer and the integer of the tabulated
  // lambda are equals
  // if (result.getP() < l_cst.getElementType().cast<IntegerType>().getWidth()) {
  //   op.emitOpError() << "should have the width of the constants less or equals "
  //                       "than the precision of the encrypted integer";
  //   return mlir::failure();
  // }
  return mlir::success();
}

} // namespace MidLFHE
} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"

#include "mlir/IR/Region.h"

#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace mlir {
namespace zamalang {
using ::mlir::zamalang::MidLFHE::AddPlainOp;
using ::mlir::zamalang::MidLFHE::GLWECipherTextType;
using ::mlir::zamalang::MidLFHE::HAddOp;
using ::mlir::zamalang::MidLFHE::MulPlainOp;

bool predPBSRegion(::mlir::Region &region) {
  if (region.getBlocks().size() != 1) {
    return false;
  }
  auto args = region.getBlocks().front().getArguments();
  if (args.size() != 1) {
    return false;
  }
  return args.front().getType().isa<mlir::IntegerType>();
}

void emitOpErrorForIncompatibleGLWEParameter(::mlir::OpState &op,
                                             ::llvm::Twine parameter) {
  ::llvm::Twine msg("should have the same GLWE ");
  op.emitError(msg.concat(parameter).concat(" parameter"));
}

bool verifyAddResultPadding(::mlir::OpState &op, GLWECipherTextType &in,
                            GLWECipherTextType &out) {
  // If the input has no value of paddingBits that doesn't constraint output.
  if (in.getPaddingBits() == -1) {
    return true;
  }
  // If the input has 0 paddingBits the ouput should have 0 paddingBits
  if (in.getPaddingBits() == 0) {
    if (out.getPaddingBits() != 0) {
      op.emitError(
          "the result shoud have 0 paddingBits has input has 0 paddingBits");
      return false;
    }
    return true;
  }
  if (in.getPaddingBits() != out.getPaddingBits() + 1) {
    op.emitError("the result should have one less padding bit than the input");
    return false;
  }
  return true;
}

bool verifyAddResultHasSameParameters(::mlir::OpState &op,
                                      GLWECipherTextType &in,
                                      GLWECipherTextType &out) {
  if (in.getDimension() != out.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return false;
  }
  if (in.getPolynomialSize() != out.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return false;
  }
  if (in.getBits() != out.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return false;
  }
  if (in.getP() != out.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return false;
  }
  if (in.getPhantomBits() != out.getPhantomBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "phantomBits");
    return false;
  }
  if (in.getScalingFactor() != out.getScalingFactor()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "scalingFactor");
    return false;
  }
  if (in.getLog2StdDev() != out.getLog2StdDev()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "log2StdDev");
    return false;
  }
  return true;
}

/**
 * veriffyAddPlainOp verify for AddPlainOp(a, x) if the GLWE parameters of the
 * output follow the rules:
 * - paddingBits:
 *    - if a.paddingBits == 0 then result.paddingBits == 0
 *    - if a.paddingBits > 0 then result.paddingBits == a.paddingBits -1
 * - every other parameters of a and the result should be equals
 */
::mlir::LogicalResult verifyAddPlainOp(AddPlainOp &op) {
  GLWECipherTextType in = op.a().getType().cast<GLWECipherTextType>();
  GLWECipherTextType out = op.getResult().getType().cast<GLWECipherTextType>();
  if (!verifyAddResultPadding(op, in, out)) {
    return ::mlir::failure();
  }
  if (!verifyAddResultHasSameParameters(op, in, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

bool verifyHAddResultPadding(::mlir::OpState &op, GLWECipherTextType &inA,
                             GLWECipherTextType &inB, GLWECipherTextType &out) {
  // If the inputs has no value of paddingBits that doesn't constraint output.
  if (inA.getPaddingBits() == -1 && inB.getPaddingBits() == -1) {
    return true;
  }
  if (inA.getPaddingBits() != inB.getPaddingBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "padding");
    return false;
  }
  return verifyAddResultPadding(op, inA, out);
}

int hAddLog2StdDevOfResult(int a, int b) {
  long double va = std::pow(std::pow(2, a), 2);
  long double vb = std::pow(std::pow(2, b), 2);
  long double vr = va + vb;
  return std::log2(::sqrt(vr));
}

bool verifyHAddResultLog2StdDev(::mlir::OpState &op, GLWECipherTextType &inA,
                                GLWECipherTextType &inB,
                                GLWECipherTextType &out) {
  // If the inputs has no value of log2StdDev that doesn't constraint output.
  if (inA.getLog2StdDev() == -1 && inB.getLog2StdDev() == -1) {
    return true;
  }
  int expectedLog2StdDev =
      hAddLog2StdDevOfResult(inA.getLog2StdDev(), inB.getLog2StdDev());
  if (out.getLog2StdDev() != expectedLog2StdDev) {
    ::llvm::Twine msg(
        "has unexpected log2StdDev parameter of its GLWE result, expected:");
    op.emitOpError(msg.concat(::llvm::Twine(expectedLog2StdDev)));
    return false;
  }
  return true;
}

bool verifyHAddSameGLWEParameter(::mlir::OpState &op, GLWECipherTextType &inA,
                                 GLWECipherTextType &inB,
                                 GLWECipherTextType &out) {
  if (inA.getDimension() != inB.getDimension() ||
      inA.getDimension() != out.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return false;
  }
  if (inA.getPolynomialSize() != inB.getPolynomialSize() ||
      inA.getPolynomialSize() != out.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return false;
  }
  if (inA.getBits() != inB.getBits() || inA.getBits() != out.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return false;
  }
  if (inA.getP() != inB.getP() || inA.getP() != out.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return false;
  }
  if (inA.getPhantomBits() != inB.getPhantomBits() ||
      inA.getPhantomBits() != out.getPhantomBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "phantomBits");
    return false;
  }
  if (inA.getScalingFactor() != inB.getScalingFactor() ||
      inA.getScalingFactor() != out.getScalingFactor()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "scalingFactor");
    return false;
  }
  return true;
}

/**
 * verifyHAddOp verify for HAddOp(a, b) if the GLWE parameters of the
 * output follow the rules:
 * - paddingBits:
 *    - if a.paddingBits == 0 then result.paddingBits == 0
 *    - if a.paddingBits > 0 then result.paddingBits == a.paddingBits -1
 *    - a.paddingBits == b.paddingBits
 * - log2StdDev:
 *    - result.log2StdDev should be equals of the result of the noise
 * propagation formula for homomorphic addition, i.e. (variance of result ==
 * variance of a + variance of b)
 * - every other parameter should be equals
 */
::mlir::LogicalResult verifyHAddOp(HAddOp &op) {
  GLWECipherTextType inA = op.a().getType().cast<GLWECipherTextType>();
  GLWECipherTextType inB = op.b().getType().cast<GLWECipherTextType>();
  GLWECipherTextType out = op.getResult().getType().cast<GLWECipherTextType>();
  if (!verifyHAddSameGLWEParameter(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  if (!verifyHAddResultPadding(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  if (!verifyHAddResultLog2StdDev(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

bool verifyMulPlainOpPadding(::mlir::OpState &op, GLWECipherTextType &inA,
                             ::mlir::Value &inB, GLWECipherTextType &out) {
  if (inA.getPaddingBits() == -1) {
    return true;
  }
  if (inA.getPaddingBits() == 0) {
    if (out.getPaddingBits() != 0) {
      op.emitError(
          "the result shoud have 0 paddingBits has input has 0 paddingBits");
      return false;
    }
    return true;
  }
  unsigned int additionalBits = 0;
  ::mlir::ConstantIntOp constantOp = inB.getDefiningOp<::mlir::ConstantIntOp>();
  if (constantOp != nullptr) {
    int64_t value = constantOp.getValue();
    additionalBits = std::ceil(std::log2(value)) + 1;
  } else {
    ::mlir::IntegerType tyB = inB.getType().cast<::mlir::IntegerType>();
    additionalBits = tyB.getIntOrFloatBitWidth();
  }
  unsigned int expectedPadding = inA.getPaddingBits() - additionalBits;
  if (out.getPaddingBits() != expectedPadding) {
    ::llvm::Twine msg(
        "has unexpected padding parameter of its GLWE result, expected:");
    op.emitOpError(msg.concat(::llvm::Twine(expectedPadding)));
    return false;
  }
  return true;
}

bool verifyMulPlainResultHasSameParameters(::mlir::OpState &op,
                                           GLWECipherTextType &in,
                                           GLWECipherTextType &out) {
  if (in.getDimension() != out.getDimension()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "dimension");
    return false;
  }
  if (in.getPolynomialSize() != out.getPolynomialSize()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "polynomialSize");
    return false;
  }
  if (in.getBits() != out.getBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "bits");
    return false;
  }
  if (in.getP() != out.getP()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "p");
    return false;
  }
  if (in.getPhantomBits() != out.getPhantomBits()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "phantomBits");
    return false;
  }
  if (in.getScalingFactor() != out.getScalingFactor()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "scalingFactor");
    return false;
  }
  if (in.getLog2StdDev() != out.getLog2StdDev()) {
    emitOpErrorForIncompatibleGLWEParameter(op, "log2StdDev");
    return false;
  }
  return true;
}

/**
 * verifyMulPlainOp verify for MulPlainOp(a, b) if the GLWE parameters of the
 * output follow the rules:
 * - paddingBits:
 *    - if a.paddingBits == 0 then result.paddingBits == 0
 *    - if a.paddingBits > 0 then result.paddingBits == a.paddingBits - log2(b)
 * - every other parameter of a and result should be equals
 */
::mlir::LogicalResult verifyMulPlainOp(MulPlainOp &op) {
  GLWECipherTextType inA = op.a().getType().cast<GLWECipherTextType>();
  ::mlir::Value inB = op.b();
  GLWECipherTextType out = op.getResult().getType().cast<GLWECipherTextType>();
  if (!verifyMulPlainOpPadding(op, inA, inB, out)) {
    return ::mlir::failure();
  }
  if (!verifyMulPlainResultHasSameParameters(op, inA, out)) {
    return ::mlir::failure();
  }
  return ::mlir::success();
}

} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"

#include "mlir/IR/Region.h"

#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

namespace mlir {
namespace zamalang {
using ::mlir::zamalang::MidLFHE::AddPlainOp;
using ::mlir::zamalang::MidLFHE::GLWECipherTextType;

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

} // namespace zamalang
} // namespace mlir

#define GET_OP_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"

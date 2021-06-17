#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.cpp.inc"

using namespace mlir::zamalang::MidLFHE;

void MidLFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.cpp.inc"
      >();
}

::mlir::Type MidLFHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  if (parser.parseOptionalKeyword("glwe").succeeded())
    return GLWECipherTextType::parse(this->getContext(), parser);
  parser.emitError(parser.getCurrentLocation(), "Unknown MidLFHE type");
  return ::mlir::Type();
}

void MidLFHEDialect::printType(::mlir::Type type,
                               ::mlir::DialectAsmPrinter &printer) const {
  mlir::zamalang::MidLFHE::GLWECipherTextType glwe =
      type.dyn_cast_or_null<mlir::zamalang::MidLFHE::GLWECipherTextType>();
  if (glwe != nullptr) {
    glwe.print(printer);
    return;
  }
  // TODO - What should be done here?
  printer << "unknwontype";
}

/*
 * Returns the nubmer of bits from the log2StdDev parameter of a GLWE.
 * This is widely inspired by the rust version on concrete:
 *
 * pub fn nb_bit_from_variance_99(var: f64, torus_bit: usize) -> usize {
 *     // compute sigma
 *     let sigma: f64 = f64::sqrt(var);
 *
 *     // the constant to get 99% of the normal distribution
 *     let z: f64 = 3.;
 *     let tmp = torus_bit as f64 + f64::log2(sigma * z);
 *     if tmp < 0. {
 *         // means no bits are affected by the noise in the integer
 * representation (discrete space) 0usize } else { tmp.ceil() as usize
 *     }
 * }
 */
unsigned nbBitsFromLog2StdDev(signed log2StdDev, signed bits) {
  long double sigma = std::pow(2, log2StdDev);
  long double z = 3;
  long double tmp = bits + std::log2(sigma * z);
  if (tmp < 0.) {
    return 0;
  }
  return std::ceil(tmp);
}

/**
 * Verify that GLWE parameter are consistant, the layout of the ciphertext is
 * organized like that.
 *
 * [0 0 0 0 0 0 0 0 X X X X X X X M M M M M M M X X X X X X X X 0 0 0 0 0 0 0 E
 * E E E E E E E E E E E E] ^ paddingBits ^               ^      p    ^ ^
 * phantomBits ^               ^ nb_bits of log2StdDev ^ ^ scalingFactor We
 * verify :
 * - The bits parameter is 32 or 64 (we support only this value for now)
 * - The message is not overlaped by the error
 * - The message is still in the ciphertext
 */
::mlir::LogicalResult GLWECipherTextType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    signed dimension, signed polynomialSize, signed bits, signed paddingBits,
    signed p, signed phantomBits, signed scalingFactor, signed log2StdDev) {
  if (bits != -1 && bits != 32 && bits != 64) {
    emitError() << "GLWE bits parameter can only be 32 or 64";
    return ::mlir::failure();
  }
  if (bits != -1 && log2StdDev != -1 && scalingFactor != -1 &&
      phantomBits != -1) {
    unsigned errBits = nbBitsFromLog2StdDev(log2StdDev, bits);
    if (errBits > scalingFactor + phantomBits) {
      emitError() << "GLWE error overlap message, errBits(" << errBits
                  << ") > scalingFactor(" << scalingFactor << ") + phantomBits("
                  << phantomBits << ")";
      return ::mlir::failure();
    }
  }
  if (bits != -1 && paddingBits != -1 && p != -1 && phantomBits != -1 &&
      scalingFactor != -1) {
    signed int phantomLeft =
        (bits - scalingFactor) - phantomBits - p - paddingBits;
    if (phantomLeft < 0) {
      emitError() << "GLWE padding + message + phantom = "
                  << phantomBits + p + paddingBits
                  << " cannot be represented  cannot be represented in bits - "
                     "scalingFactor = "
                  << (bits - scalingFactor);
      return ::mlir::failure();
    }
  }
  return ::mlir::success();
}

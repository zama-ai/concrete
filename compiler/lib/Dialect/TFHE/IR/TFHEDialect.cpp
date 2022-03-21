// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOpsTypes.cpp.inc"

#include "concretelang/Dialect/TFHE/IR/TFHEOpsDialect.cpp.inc"

#include "concretelang/Support/Constants.h"

using namespace mlir::concretelang::TFHE;

void TFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/TFHE/IR/TFHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/TFHE/IR/TFHEOpsTypes.cpp.inc"
      >();
}

::mlir::Type TFHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  if (parser.parseOptionalKeyword("glwe").succeeded())
    return GLWECipherTextType::parse(parser);
  parser.emitError(parser.getCurrentLocation(), "Unknown TFHE type");
  return ::mlir::Type();
}

void TFHEDialect::printType(::mlir::Type type,
                            ::mlir::DialectAsmPrinter &printer) const {
  mlir::concretelang::TFHE::GLWECipherTextType glwe =
      type.dyn_cast_or_null<mlir::concretelang::TFHE::GLWECipherTextType>();
  if (glwe != nullptr) {
    glwe.print(printer);
    return;
  }
  // TODO - What should be done here?
  printer << "unknwontype";
}

/// Verify that GLWE parameter are consistant
/// - The bits parameter is 64 (we support only this for v0)
::mlir::LogicalResult GLWECipherTextType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    signed dimension, signed polynomialSize, signed bits, signed p) {
  if (bits != -1 && bits != 64) {
    emitError() << "GLWE bits parameter can only be 64";
    return ::mlir::failure();
  }
  if (p == 0) {
    emitError() << "GLWE p parameter must be positive";
    return mlir::failure();
  }
  return ::mlir::success();
}

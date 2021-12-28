// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "concretelang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "concretelang/Dialect/MidLFHE/IR/MidLFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.cpp.inc"

#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOpsDialect.cpp.inc"

using namespace mlir::concretelang::MidLFHE;

void MidLFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.cpp.inc"
      >();
}

::mlir::Type MidLFHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  if (parser.parseOptionalKeyword("glwe").succeeded())
    return GLWECipherTextType::parse(parser);
  parser.emitError(parser.getCurrentLocation(), "Unknown MidLFHE type");
  return ::mlir::Type();
}

void MidLFHEDialect::printType(::mlir::Type type,
                               ::mlir::DialectAsmPrinter &printer) const {
  mlir::concretelang::MidLFHE::GLWECipherTextType glwe =
      type.dyn_cast_or_null<mlir::concretelang::MidLFHE::GLWECipherTextType>();
  if (glwe != nullptr) {
    glwe.print(printer);
    return;
  }
  // TODO - What should be done here?
  printer << "unknwontype";
}

/// Verify that GLWE parameter are consistant
/// - The bits parameter is 64 (we support only this for v0)
/// - The p parameter is ]0;7]
::mlir::LogicalResult GLWECipherTextType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    signed dimension, signed polynomialSize, signed bits, signed p) {
  if (bits != -1 && bits != 64) {
    emitError() << "GLWE bits parameter can only be 64";
    return ::mlir::failure();
  }
  if (p == 0 || p > 7) {
    emitError() << "GLWE p parameter can only be in ]0;7]";
    return mlir::failure();
  }
  return ::mlir::success();
}

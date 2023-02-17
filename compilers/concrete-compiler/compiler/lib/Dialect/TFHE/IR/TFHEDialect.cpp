// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.cpp.inc"

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

  addAttributes<
#define GET_ATTRDEF_LIST
#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.cpp.inc"
      >();
}

/// Verify that GLWE parameter are consistant
/// - The bits parameter is 64 (we support only this for v0)
::mlir::LogicalResult GLWECipherTextType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    GLWESecretKey key) {
  if (!key.isNotParameterized() && key.getPolySize().value() == 0) {
    emitError() << "GLWE key has zero poly size.";
    return ::mlir::failure();
  }
  if (!key.isNotParameterized() && key.getDimension().value() == 0) {
    emitError() << "GLWE key has zero dimension.";
    return ::mlir::failure();
  }
  return ::mlir::success();
}

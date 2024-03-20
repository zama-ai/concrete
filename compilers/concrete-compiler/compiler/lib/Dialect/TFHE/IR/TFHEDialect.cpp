// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/TFHE/IR/TFHEDialect.h"
#include "concretelang/Dialect/TFHE/IR/TFHEOps.h"
#include "concretelang/Dialect/TFHE/IR/TFHEParameters.h"

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/TFHE/IR/TFHEOpsTypes.cpp.inc"

#include "concretelang/Dialect/TFHE/IR/TFHEOpsDialect.cpp.inc"

#include "concretelang/Support/Constants.h"
#include "concretelang/Support/Variants.h"

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

/// Verify that GLWE parameter are consistent
::mlir::LogicalResult GLWECipherTextType::verify(
    ::llvm::function_ref<::mlir::InFlightDiagnostic()> emitError,
    GLWESecretKey key) {
  return std::visit(
      overloaded{[](GLWESecretKeyNone sk) { return mlir::success(); },
                 [&](GLWESecretKeyParameterized sk) {
                   if (sk.dimension == 0) {
                     emitError() << "GLWE key has zero dimension.";
                     return ::mlir::failure();
                   }
                   if (sk.polySize == 0) {
                     emitError() << "GLWE key has zero poly size.";
                     return ::mlir::failure();
                   }
                   return mlir::success();
                 },
                 [&](GLWESecretKeyNormalized sk) {
                   if (sk.dimension == 0) {
                     emitError() << "GLWE key has zero dimension.";
                     return ::mlir::failure();
                   }
                   if (sk.polySize == 0) {
                     emitError() << "GLWE key has zero poly size.";
                     return ::mlir::failure();
                   }
                   return mlir::success();
                 }},
      key.inner);
}

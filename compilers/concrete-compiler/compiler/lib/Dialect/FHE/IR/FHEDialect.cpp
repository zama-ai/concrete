// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"
#include "concretelang/Dialect/FHE/Interfaces/FHEInterfaces.h"

#define GET_ATTRDEF_CLASSES
#include "concretelang/Dialect/FHE/IR/FHEAttrs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/FHE/IR/FHEOpsTypes.cpp.inc"

#include "concretelang/Dialect/FHE/IR/FHEOpsDialect.cpp.inc"

#include "concretelang/Support/Constants.h"

using namespace mlir::concretelang::FHE;

void FHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/FHE/IR/FHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/FHE/IR/FHEOpsTypes.cpp.inc"
      >();

  mlir::Dialect::addAttributes<
#define GET_ATTRDEF_LIST
#include "concretelang/Dialect/FHE/IR/FHEAttrs.cpp.inc"
      >();
}

mlir::LogicalResult EncryptedUnsignedIntegerType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned p) {
  if (p == 0) {
    emitError() << "FHE.eint doesn't support precision of 0";
    return mlir::failure();
  }
  return mlir::success();
}

void EncryptedUnsignedIntegerType::print(mlir::AsmPrinter &p) const {
  p << "<" << getWidth() << ">";
}

mlir::Type EncryptedUnsignedIntegerType::parse(mlir::AsmParser &p) {
  if (p.parseLess())
    return mlir::Type();

  int width;

  if (p.parseInteger(width))
    return mlir::Type();

  if (p.parseGreater())
    return mlir::Type();

  mlir::Location loc = p.getEncodedSourceLoc(p.getNameLoc());

  return getChecked(loc, loc.getContext(), width);
}

mlir::LogicalResult EncryptedSignedIntegerType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned p) {
  if (p == 0) {
    emitError() << "FHE.esint doesn't support precision of 0";
    return mlir::failure();
  }
  return mlir::success();
}

void EncryptedSignedIntegerType::print(mlir::AsmPrinter &p) const {
  p << "<" << getWidth() << ">";
}

mlir::Type EncryptedSignedIntegerType::parse(mlir::AsmParser &p) {
  if (p.parseLess())
    return mlir::Type();

  int width;

  if (p.parseInteger(width))
    return mlir::Type();

  if (p.parseGreater())
    return mlir::Type();

  mlir::Location loc = p.getEncodedSourceLoc(p.getNameLoc());

  return getChecked(loc, loc.getContext(), width);
}

// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/Dialect/FHE/IR/FHEDialect.h"
#include "concretelang/Dialect/FHE/IR/FHEOps.h"
#include "concretelang/Dialect/FHE/IR/FHETypes.h"

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
}

::mlir::Type FHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  mlir::Type type;

  if (parser.parseOptionalKeyword("eint").succeeded()) {
    generatedTypeParser(parser, "eint", type);
    return type;
  }

  // TODO
  // Don't have a parser for a custom type
  // We shouldn't call the default parser
  // but what should we do instead?
  parser.parseType(type);
  return type;
}

void FHEDialect::printType(::mlir::Type type,
                           ::mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).failed())
    // Calling default printer if failed to print FHE type
    printer.printType(type);
}

mlir::LogicalResult EncryptedIntegerType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned p) {
  if (p == 0) {
    emitError() << "FHE.eint didn't support precision equals to 0";
    return mlir::failure();
  }
  return mlir::success();
}

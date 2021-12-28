// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "concretelang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "concretelang/Dialect/HLFHE/IR/HLFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/HLFHE/IR/HLFHEOpsTypes.cpp.inc"

#include "concretelang/Dialect/HLFHE/IR/HLFHEOpsDialect.cpp.inc"

using namespace mlir::concretelang::HLFHE;

void HLFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/HLFHE/IR/HLFHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/HLFHE/IR/HLFHEOpsTypes.cpp.inc"
      >();
}

::mlir::Type HLFHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
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

void HLFHEDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).failed())
    // Calling default printer if failed to print HLFHE type
    printer.printType(type);
}

mlir::LogicalResult EncryptedIntegerType::verify(
    llvm::function_ref<::mlir::InFlightDiagnostic()> emitError, unsigned p) {
  if (p == 0 || p > 7) {
    emitError() << "HLFHE.eint support only precision in ]0;7]";
    return mlir::failure();
  }
  return mlir::success();
}

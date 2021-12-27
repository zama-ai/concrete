// Part of the Concrete Compiler Project, under the BSD3 License with Zama Exceptions.
// See https://github.com/zama-ai/homomorphizer/blob/master/LICENSE.txt for license information.

#include "zamalang/Dialect/LowLFHE/IR/LowLFHEDialect.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOps.h"
#include "zamalang/Dialect/LowLFHE/IR/LowLFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOpsTypes.cpp.inc"

#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOpsDialect.cpp.inc"

using namespace mlir::zamalang::LowLFHE;

void LowLFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zamalang/Dialect/LowLFHE/IR/LowLFHEOpsTypes.cpp.inc"
      >();
}

::mlir::Type LowLFHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  mlir::Type type;

  std::string types_str[] = {
      "plaintext",       "plaintext_list",     "foreign_plaintext_list",
      "lwe_ciphertext",  "lwe_key_switch_key", "lwe_bootstrap_key",
      "glwe_ciphertext", "cleartext",          "context",
  };

  for (const std::string &type_str : types_str) {
    if (parser.parseOptionalKeyword(type_str).succeeded()) {
      generatedTypeParser(parser, type_str, type);
      return type;
    }
  }

  parser.emitError(parser.getCurrentLocation(), "Unknown LowLFHE type");

  return type;
}

void LowLFHEDialect::printType(::mlir::Type type,
                               ::mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).failed())
    // Calling default printer if failed to print LowLFHE type
    printer.printType(type);
}

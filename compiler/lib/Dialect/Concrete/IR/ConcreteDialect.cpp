// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete-compiler-internal/blob/master/LICENSE.txt
// for license information.

#include "concretelang/Dialect/Concrete/IR/ConcreteDialect.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.h"
#include "concretelang/Dialect/Concrete/IR/ConcreteTypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/Concrete/IR/ConcreteOpsTypes.cpp.inc"

#include "concretelang/Dialect/Concrete/IR/ConcreteOpsDialect.cpp.inc"

using namespace mlir::concretelang::Concrete;

void ConcreteDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/Concrete/IR/ConcreteOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/Concrete/IR/ConcreteOpsTypes.cpp.inc"
      >();
}

::mlir::Type
ConcreteDialect::parseType(::mlir::DialectAsmParser &parser) const {
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

  parser.emitError(parser.getCurrentLocation(), "Unknown Concrete type");

  return type;
}

void ConcreteDialect::printType(::mlir::Type type,
                                ::mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).failed())
    // Calling default printer if failed to print Concrete type
    printer.printType(type);
}

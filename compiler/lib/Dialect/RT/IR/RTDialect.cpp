#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "zamalang/Dialect/RT/IR/RTDialect.h"
#include "zamalang/Dialect/RT/IR/RTOps.h"
#include "zamalang/Dialect/RT/IR/RTTypes.h"

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/RT/IR/RTOpsTypes.cpp.inc"

#include "zamalang/Dialect/RT/IR/RTOpsDialect.cpp.inc"

using namespace mlir::zamalang::RT;

void RTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zamalang/Dialect/RT/IR/RTOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zamalang/Dialect/RT/IR/RTOpsTypes.cpp.inc"
      >();
}

::mlir::Type RTDialect::parseType(::mlir::DialectAsmParser &parser) const {
  mlir::Type type;
  if (parser.parseOptionalKeyword("future").succeeded()) {
    generatedTypeParser(parser, "future", type);
    return type;
  }
  return type;
}

void RTDialect::printType(::mlir::Type type,
                          ::mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).failed()) {
    printer.printType(type);
  }
}

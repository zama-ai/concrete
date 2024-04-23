// Part of the Concrete Compiler Project, under the BSD3 License with Zama
// Exceptions. See
// https://github.com/zama-ai/concrete/blob/main/LICENSE.txt
// for license information.

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/FunctionImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#include "concretelang/Dialect/RT/IR/RTDialect.h"
#include "concretelang/Dialect/RT/IR/RTOps.h"
#include "concretelang/Dialect/RT/IR/RTTypes.h"

#define GET_TYPEDEF_CLASSES
#include "concretelang/Dialect/RT/IR/RTOpsTypes.cpp.inc"

#include "concretelang/Dialect/RT/IR/RTOpsDialect.cpp.inc"

using namespace mlir::concretelang::RT;

void RTDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "concretelang/Dialect/RT/IR/RTOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "concretelang/Dialect/RT/IR/RTOpsTypes.cpp.inc"
      >();
}

::mlir::Type RTDialect::parseType(::mlir::DialectAsmParser &parser) const {
  mlir::Type type;
  llvm::StringRef mnenomic;

  generatedTypeParser(parser, &mnenomic, type);

  return type;
}

void RTDialect::printType(::mlir::Type type,
                          ::mlir::DialectAsmPrinter &printer) const {
  if (generatedTypePrinter(type, printer).failed()) {
    printer.printType(type);
  }
}

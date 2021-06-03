#include "zamalang/Dialect/HLFHE/IR/HLFHEDialect.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.h"
#include "zamalang/Dialect/HLFHE/IR/HLFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/HLFHE/IR/HLFHEOpsTypes.cpp.inc"

using namespace mlir::zamalang::HLFHE;

void HLFHEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "zamalang/Dialect/HLFHE/IR/HLFHEOps.cpp.inc"
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "zamalang/Dialect/HLFHE/IR/HLFHEOpsTypes.cpp.inc"
      >();
}

::mlir::Type HLFHEDialect::parseType(::mlir::DialectAsmParser &parser) const {
  if (parser.parseKeyword("eint").failed())
    return ::mlir::Type();

  return EncryptedIntegerType::parse(this->getContext(), parser);
}

void HLFHEDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &printer) const {
  mlir::zamalang::HLFHE::EncryptedIntegerType eint =
      type.dyn_cast_or_null<mlir::zamalang::HLFHE::EncryptedIntegerType>();
  if (eint != nullptr) {
    eint.print(printer);
    return;
  }
  // TODO - What should be done here?
  printer << "unknwontype";
}
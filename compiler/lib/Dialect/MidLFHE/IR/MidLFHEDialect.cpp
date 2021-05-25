#include "zamalang/Dialect/MidLFHE/IR/MidLFHEDialect.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.h"
#include "zamalang/Dialect/MidLFHE/IR/MidLFHETypes.h"

#define GET_TYPEDEF_CLASSES
#include "zamalang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.cpp.inc"

using namespace mlir::zamalang::MidLFHE;

void MidLFHEDialect::initialize() {
  addOperations<
    #define GET_OP_LIST
    #include "zamalang/Dialect/MidLFHE/IR/MidLFHEOps.cpp.inc"
      >();

  addTypes<
    #define GET_TYPEDEF_LIST
    #include "zamalang/Dialect/MidLFHE/IR/MidLFHEOpsTypes.cpp.inc"
  >();
}

::mlir::Type MidLFHEDialect::parseType(::mlir::DialectAsmParser &parser) const
{
  if(parser.parseOptionalKeyword("lwe").succeeded())
    return LWECipherTextType::parse(this->getContext(), parser);
  if(parser.parseOptionalKeyword("glwe").succeeded())
    return GLWECipherTextType::parse(this->getContext(), parser);
  if(parser.parseOptionalKeyword("ggsw").succeeded())
    return GGSWCipherTextType::parse(this->getContext(), parser);
  if(parser.parseOptionalKeyword("ciphertext").succeeded())
    return AnyCipherTextType::parse(this->getContext(), parser);
  parser.emitError(parser.getCurrentLocation(), "Unknown MidLFHE type");
  return ::mlir::Type();
}

void MidLFHEDialect::printType(::mlir::Type type,
                             ::mlir::DialectAsmPrinter &printer) const
{
  mlir::zamalang::MidLFHE::LWECipherTextType lwe = type.dyn_cast_or_null<mlir::zamalang::MidLFHE::LWECipherTextType>();
  if (lwe != nullptr) {
    lwe.print(printer);
    return;
  }
  mlir::zamalang::MidLFHE::GLWECipherTextType glwe = type.dyn_cast_or_null<mlir::zamalang::MidLFHE::GLWECipherTextType>();
  if (glwe != nullptr) {
    glwe.print(printer);
    return;
  }
  mlir::zamalang::MidLFHE::GGSWCipherTextType ggsw = type.dyn_cast_or_null<mlir::zamalang::MidLFHE::GGSWCipherTextType>();
  if (ggsw != nullptr) {
    ggsw.print(printer);
    return;
  }
  mlir::zamalang::MidLFHE::AnyCipherTextType any = type.dyn_cast_or_null<mlir::zamalang::MidLFHE::AnyCipherTextType>();
  if (any != nullptr) {
    any.print(printer);
    return;
  }
  // TODO - What should be done here?
  printer << "unknwontype";
}
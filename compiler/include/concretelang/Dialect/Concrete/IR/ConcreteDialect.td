#ifndef CONCRETELANG_DIALECT_Concrete_IR_Concrete_DIALECT
#define CONCRETELANG_DIALECT_Concrete_IR_Concrete_DIALECT

include "mlir/IR/OpBase.td"

def Concrete_Dialect : Dialect {
  let name = "Concrete";
  let summary = "Low Level Fully Homorphic Encryption dialect";
  let description = [{
    A dialect for representation of low level operation on fully homomorphic ciphertext.
  }];
  let cppNamespace = "::mlir::concretelang::Concrete";
  let useDefaultTypePrinterParser = 0;
  let extraClassDeclaration = [{
    ::mlir::Type parseType(::mlir::DialectAsmParser &parser) const override;
    void printType(::mlir::Type type, ::mlir::DialectAsmPrinter &printer) const override;
  }];
}

#endif

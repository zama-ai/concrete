#ifndef CONCRETELANG_DIALECT_FHE_IR_FHE_INTERFACES
#define CONCRETELANG_DIALECT_FHE_IR_FHE_INTERFACES

include "mlir/IR/OpBase.td"

def FheIntegerInterface : TypeInterface<"FheIntegerInterface"> {
  let cppNamespace = "mlir::concretelang::FHE";

  let description = [{
    Interface for encapsulating the common properties of encrypted integer types.
  }];

  let methods = [
    InterfaceMethod<
      /*description=*/"Get bit-width of the integer.",
      /*retTy=*/"unsigned",
      /*methodName=*/"getWidth"
    >,
    InterfaceMethod<
      /*description=*/"Get whether the integer is signed.",
      /*retTy=*/"bool",
      /*methodName=*/"isSigned"
    >,
    InterfaceMethod<
      /*description=*/"Get whether the integer is unsigned.",
      /*retTy=*/"bool",
      /*methodName=*/"isUnsigned"
    >
  ];
}

#endif

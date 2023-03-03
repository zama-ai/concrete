#ifndef CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_TYPES
#define CONCRETELANG_DIALECT_FHELinalg_IR_FHELinalg_TYPES

include "concretelang/Dialect/FHELinalg/IR/FHELinalgDialect.td"
include "mlir/IR/BuiltinTypes.td"
include "concretelang/Dialect/FHE/IR/FHETypes.td"

class FHELinalg_Type<string name, list<Trait> traits = []> :
  TypeDef<FHELinalg_Dialect, name, traits> { }

#endif

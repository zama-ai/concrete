#ifndef CONCRETELANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_TYPES
#define CONCRETELANG_DIALECT_HLFHELinalg_IR_HLFHELinalg_TYPES

include "concretelang/Dialect/HLFHELinalg/IR/HLFHELinalgDialect.td"
include "mlir/IR/BuiltinTypes.td"
include "concretelang/Dialect/HLFHE/IR/HLFHETypes.td"

class HLFHELinalg_Type<string name, list<Trait> traits = []> :
  TypeDef<HLFHELinalg_Dialect, name, traits> { }

#endif

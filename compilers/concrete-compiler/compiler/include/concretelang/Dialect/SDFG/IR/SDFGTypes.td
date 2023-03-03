#ifndef CONCRETELANG_DIALECT_SDFG_IR_SDFG_TYPES
#define CONCRETELANG_DIALECT_SDFG_IR_SDFG_TYPES

include "concretelang/Dialect/SDFG/IR/SDFGDialect.td"
include "mlir/IR/BuiltinTypes.td"

class SDFG_Type<string name, list<Trait> traits = []> :
  TypeDef<SDFG_Dialect, name, traits> { }

def SDFG_DFG : SDFG_Type<"DFG", []> {
  let mnemonic = "dfg";

  let summary = "An SDFG data flow graph";

  let description = [{
    A handle to an SDFG data flow graph
  }];

  let parameters = (ins);
  let hasCustomAssemblyFormat = 0;
}


def SDFG_Stream : SDFG_Type<"Stream", []> {
  let mnemonic = "stream";

  let summary = "An SDFG data stream";

  let description = [{
    An SDFG stream to connect SDFG processes.
  }];

  let parameters = (ins "Type":$elementType);
  let hasCustomAssemblyFormat = 1;
}

#endif

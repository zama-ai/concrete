#ifndef CONCRETELANG_FHELINALG_TILING_PASS
#define CONCRETELANG_FHELINALG_TILING_PASS

include "mlir/Pass/PassBase.td"

def FinalizingBufferize : FunctionPass<"concretelang-bufferize"> {
  let summary =
      "Marks FHELinalg operations for tiling using a vector of tile sizes";
  let constructor = "mlir::concretelang::createBufferizePass()";
  let options = [];
  let dependentDialects = [ "mlir::memref::MemRefDialect" ];
}

#endif

#ifndef CONCRETELANG_HLFHELINALG_TILING_PASS
#define CONCRETELANG_HLFHELINALG_TILING_PASS

include "mlir/Pass/PassBase.td"

def HLFHELinalgTilingMarker : Pass<"hlfhe-linalg-tiling-marker"> {
  let summary =
      "Marks HLFHELinalg operations for tiling using a vector of tile sizes";
  let constructor = "mlir::concretelang::createHLFHELinalgTilingMarkerPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::HLFHELinalg::HLFHELinalgDialect" ];
}

def HLFHELinalgTiling : Pass<"hlfhe-linalg-tiling"> {
  let summary = "Performs tiling of HLFHELinalg operations based on the "
                "tile-size attribute";
  let constructor = "mlir::concretelang::createHLFHELinalgTilingPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::HLFHELinalg::HLFHELinalgDialect" ];
}

#endif

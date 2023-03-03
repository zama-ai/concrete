#ifndef CONCRETELANG_FHELINALG_TILING_PASS
#define CONCRETELANG_FHELINALG_TILING_PASS

include "mlir/Pass/PassBase.td"

def FHELinalgTilingMarker : Pass<"fhe-linalg-tiling-marker"> {
  let summary =
      "Marks FHELinalg operations for tiling using a vector of tile sizes";
  let constructor = "mlir::concretelang::createFHELinalgTilingMarkerPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::FHELinalg::FHELinalgDialect" ];
}

def FHELinalgTiling : Pass<"fhe-linalg-tiling"> {
  let summary = "Performs tiling of FHELinalg operations based on the "
                "tile-size attribute";
  let constructor = "mlir::concretelang::createFHELinalgTilingPass()";
  let options = [];
  let dependentDialects = [ "mlir::concretelang::FHELinalg::FHELinalgDialect" ];
}

#endif

#ifndef ZAMALANG_HLFHELINALG_TILING_PASS
#define ZAMALANG_HLFHELINALG_TILING_PASS

include "mlir/Pass/PassBase.td"

def HLFHELinalgTilingMarker : Pass<"hlfhe-linalg-tiling-marker"> {
  let summary =
      "Marks HLFHELinalg operations for tiling using a vector of tile sizes";
  let constructor = "mlir::zamalang::createHLFHELinalgTilingMarkerPass()";
  let options = [];
  let dependentDialects = [ "mlir::zamalang::HLFHELinalg::HLFHELinalgDialect" ];
}

def HLFHELinalgTiling : Pass<"hlfhe-linalg-tiling"> {
  let summary = "Performs tiling of HLFHELinalg operations based on the "
                "tile-size attribute";
  let constructor = "mlir::zamalang::createHLFHELinalgTilingPass()";
  let options = [];
  let dependentDialects = [ "mlir::zamalang::HLFHELinalg::HLFHELinalgDialect" ];
}

#endif

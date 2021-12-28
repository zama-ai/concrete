// RUN: concretecompiler --split-input-file --verify-diagnostics --action=roundtrip %s

// Bad dimension of the lookup table
func @apply_lookup_table(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: tensor<4xi2>) -> !MidLFHE.glwe<{512,10,64}{2}> {
  // expected-error @+1 {{'MidLFHE.apply_lookup_table' op : `l_cst` (operand #2) inner dimension should have size 128(=2^7) to match `ct` (operand #1) elements bitwidth (7)}}  
  %1 = "MidLFHE.apply_lookup_table"(%arg0, %arg1) {glweDimension = 1 : i32, polynomialSize = 1024 : i32, levelKS = 2 : i32, baseLogKS = -82 : i32, levelBS = 3 : i32, baseLogBS = -83 : i32, outputSizeKS = 600 : i32}: (!MidLFHE.glwe<{1024,12,64}{7}>, tensor<4xi2>) -> (!MidLFHE.glwe<{512,10,64}{2}>)
  return %1: !MidLFHE.glwe<{512,10,64}{2}>
}


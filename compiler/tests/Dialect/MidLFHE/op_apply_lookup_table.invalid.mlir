// RUN: zamacompiler --split-input-file --verify-diagnostics %s

// Bad dimension of the lookup table
func @apply_lookup_table(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: memref<4xi2>) -> !MidLFHE.glwe<{512,10,64}{2}> {
  // expected-error @+1 {{'MidLFHE.apply_lookup_table' op should have as `l_cst` argument a shape of one dimension equals to 2^p, where p is the width of the `ct` argument}}
  %1 = "MidLFHE.apply_lookup_table"(%arg0, %arg1) {k = 1 : i32, polynomialSize = 1024 : i32, levelKS = 2 : i32, baseLogKS = -82 : i32, levelBS = 3 : i32, baseLogBS = -83 : i32}: (!MidLFHE.glwe<{1024,12,64}{7}>, memref<4xi2>) -> (!MidLFHE.glwe<{512,10,64}{2}>)
  return %1: !MidLFHE.glwe<{512,10,64}{2}>
}

// -----

// Bad dimension of integer in the lookup table
func @apply_lookup_table(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: memref<128xi3>) -> !MidLFHE.glwe<{512,10,64}{2}> {
  // expected-error @+1 {{'MidLFHE.apply_lookup_table' op should have equals width beetwen the encrypted integer result and integers of the `tabulated_lambda` argument}}
  %1 = "MidLFHE.apply_lookup_table"(%arg0, %arg1) {k = 1 : i32, polynomialSize = 1024 : i32, levelKS = 2 : i32, baseLogKS = -82 : i32, levelBS = 3 : i32, baseLogBS = -83 : i32}: (!MidLFHE.glwe<{1024,12,64}{7}>, memref<128xi3>) -> (!MidLFHE.glwe<{512,10,64}{2}>)
  return %1: !MidLFHE.glwe<{512,10,64}{2}>
}
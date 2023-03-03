// RUN: concretecompiler %s --optimize-tfhe=false --action=dump-tfhe 2>&1| FileCheck %s

// CHECK: func.func @apply_lookup_table(%arg0: !TFHE.glwe<{_,_,_}{2}>, %arg1: tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{3}> {
// CHECK-NEXT: %0 = "TFHE.encode_expand_lut_for_bootstrap"(%arg1) {isSigned = false, outputBits = 3 : i32, polySize = 256 : i32} : (tensor<4xi64>) -> tensor<256xi64>
// CHECK-NEXT: %1 = "TFHE.keyswitch_glwe"(%arg0) {baseLog = -1 : i32, level = -1 : i32} : (!TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT: %2 = "TFHE.bootstrap_glwe"(%1, %0) {baseLog = -1 : i32, glweDimension = -1 : i32, level = -1 : i32, polySize = -1 : i32} : (!TFHE.glwe<{_,_,_}{2}>, tensor<256xi64>) -> !TFHE.glwe<{_,_,_}{3}>
// CHECK-NEXT: return %2 : !TFHE.glwe<{_,_,_}{3}>
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<3> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

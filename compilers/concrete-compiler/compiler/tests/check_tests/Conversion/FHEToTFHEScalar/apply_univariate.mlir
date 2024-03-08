// RUN: concretecompiler %s --optimize-tfhe=false --optimizer-strategy=dag-mono --action=dump-tfhe 2>&1| FileCheck %s

// CHECK: func.func @apply_lookup_table(%arg0: !TFHE.glwe<sk?>, %arg1: tensor<4xi64>) -> !TFHE.glwe<sk?> {
// CHECK-NEXT: %0 = "TFHE.encode_expand_lut_for_bootstrap"(%arg1) {isSigned = false, outputBits = 3 : i32, polySize = 256 : i32} : (tensor<4xi64>) -> tensor<256xi64>
// CHECK-NEXT: %1 = "TFHE.keyswitch_glwe"(%arg0) {key = #TFHE.ksk<sk?, sk?, -1, -1>} : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
// CHECK-NEXT: %2 = "TFHE.bootstrap_glwe"(%1, %0) {key = #TFHE.bsk<sk?, sk?, -1, -1, -1, -1>} : (!TFHE.glwe<sk?>, tensor<256xi64>) -> !TFHE.glwe<sk?>
// CHECK-NEXT: return %2 : !TFHE.glwe<sk?>
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<3> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

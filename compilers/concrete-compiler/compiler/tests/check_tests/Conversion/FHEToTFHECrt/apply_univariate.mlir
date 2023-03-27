// RUN: concretecompiler --optimize-tfhe=false --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

// CHECK: func.func @apply_lookup_table(%arg0: tensor<5x!TFHE.glwe<sk?>>, %arg1: tensor<4xi64>) -> tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT: %0 = "TFHE.encode_lut_for_crt_woppbs"(%arg1) {crtBits = [1, 2, 3, 3, 4], crtDecomposition = [2, 3, 5, 7, 11], isSigned = false, modulusProduct = 2310 : i32} : (tensor<4xi64>) -> tensor<5x8192xi64>
// CHECK-NEXT: %1 = "TFHE.wop_pbs_glwe"(%arg0, %0)  {bsk = #TFHE.bsk<sk?, sk?, -1, -1, -1, -1>, cbsBaseLog = -1 : i32, cbsLevels = -1 : i32, crtDecomposition = [], ksk = #TFHE.ksk<sk?, sk?, -1, -1>, pksk = #TFHE.pksk<sk?, sk?, -1, -1, -1, -1, -1>} : (tensor<5x!TFHE.glwe<sk?>>, tensor<5x8192xi64>) -> tensor<5x!TFHE.glwe<sk?>>
// CHECK-NEXT: return %1 : tensor<5x!TFHE.glwe<sk?>>
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<3> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

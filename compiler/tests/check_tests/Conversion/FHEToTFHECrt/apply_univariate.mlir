// RUN: concretecompiler --action=dump-tfhe %s --large-integer-crt-decomposition=2,3,5,7,11 --large-integer-circuit-bootstrap=2,9 --large-integer-packing-keyswitch=694,1024,4,9 --v0-parameter=2,10,693,4,9,7,2 2>&1| FileCheck %s

// CHECK: func.func @apply_lookup_table(%arg0: tensor<5x!TFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<4xi64>) -> tensor<5x!TFHE.glwe<{_,_,_}{3}>>
// CHECK-NEXT: %0 = "TFHE.encode_expand_lut_for_woppbs"(%arg1) {crtBits = [1, 2, 3, 3, 4], crtDecomposition = [2, 3, 5, 7, 11], isSigned = false, modulusProduct = 2310 : i32, polySize = 1024 : i32} : (tensor<4xi64>) -> tensor<40960xi64>
// CHECK-NEXT: %1 = "TFHE.wop_pbs_glwe"(%arg0, %0) {bootstrapBaseLog = -1 : i32, bootstrapLevel = -1 : i32, circuitBootstrapBaseLog = -1 : i32, circuitBootstrapLevel = -1 : i32, crtDecomposition = [], keyswitchBaseLog = -1 : i32, keyswitchLevel = -1 : i32, packingKeySwitchBaseLog = -1 : i32, packingKeySwitchInputLweDimension = -1 : i32, packingKeySwitchLevel = -1 : i32, packingKeySwitchoutputPolynomialSize = -1 : i32} : (tensor<5x!TFHE.glwe<{_,_,_}{2}>>, tensor<40960xi64>) -> tensor<5x!TFHE.glwe<{_,_,_}{3}>>
// CHECK-NEXT: return %1 : tensor<5x!TFHE.glwe<{_,_,_}{3}>>
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<3> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

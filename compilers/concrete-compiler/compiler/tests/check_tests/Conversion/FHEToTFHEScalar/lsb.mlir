// RUN: concretecompiler %s --passes=fhe-to-tfhe-scalar --v0-parameter=2,10,693,4,9,7,2 --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @lsb(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
func.func @lsb(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT:  %c128_i64 = arith.constant 128 : i64
  // CHECK-NEXT:  %0 = "TFHE.mul_glwe_int"(%arg0, %c128_i64) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  %cst = arith.constant dense<-36028797018963968> : tensor<1024xi64>
  // CHECK-NEXT:  %c4611686018427387904_i64 = arith.constant 4611686018427387904 : i64
  // CHECK-NEXT:  %1 = "TFHE.add_glwe_int"(%0, %c4611686018427387904_i64) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  %2 = "TFHE.keyswitch_glwe"(%1) {key = #TFHE.ksk<sk?, sk?, -1, -1>} : (!TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  %3 = "TFHE.bootstrap_glwe"(%2, %cst) {key = #TFHE.bsk<sk?, sk?, -1, -1, -1, -1>} : (!TFHE.glwe<sk?>, tensor<1024xi64>) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  %c36028797018963968_i64 = arith.constant 36028797018963968 : i64
  // CHECK-NEXT:  %4 = "TFHE.add_glwe_int"(%3, %c36028797018963968_i64) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  return %4 : !TFHE.glwe<sk?>
  %0 = "FHE.lsb"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %0: !FHE.eint<7>
}

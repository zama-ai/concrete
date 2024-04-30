// RUN: concretecompiler %s --passes=fhe-to-tfhe-scalar --v0-parameter=2,10,693,4,9,7,2 --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_eint_int(%arg0: !TFHE.glwe<sk?>) -> !TFHE.glwe<sk?>
func.func @add_eint_int(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT:  %c1_i8 = arith.constant 1 : i8
  // CHECK-NEXT:  %0 = arith.extsi %c1_i8 : i8 to i64
  // CHECK-NEXT:  %c56_i64 = arith.constant 56 : i64
  // CHECK-NEXT:  %1 = arith.shli %0, %c56_i64 : i64
  // CHECK-NEXT:  %2 = "TFHE.add_glwe_int"(%arg0, %1) : (!TFHE.glwe<sk?>, i64) -> !TFHE.glwe<sk?>
  // CHECK-NEXT:  return %2 : !TFHE.glwe<sk?>


  %0 = arith.constant 1 : i8
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}

// RUN: concretecompiler %s --passes fhe-to-tfhe --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func @add_eint_int(%arg0: !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
func @add_eint_int(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "TFHE.add_glwe_int"(%arg0, %[[V1]]) : (!TFHE.glwe<{_,_,_}{7}>, i8) -> !TFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT: return %[[V2]] : !TFHE.glwe<{_,_,_}{7}>

  %0 = arith.constant 1 : i8
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<7>, i8) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}

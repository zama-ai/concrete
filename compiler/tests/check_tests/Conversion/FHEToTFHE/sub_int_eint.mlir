// RUN: concretecompiler %s --passes fhe-to-tfhe --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @sub_int_eint(%arg0: !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
func.func @sub_int_eint(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "TFHE.sub_int_glwe"(%[[V1]], %arg0) : (i8, !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT: return %[[V2]] : !TFHE.glwe<{_,_,_}{7}>

  %0 = arith.constant 1 : i8
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i8, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}

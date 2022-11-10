// RUN: concretecompiler %s --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_eint(%arg0: !TFHE.glwe<{_,_,_}{7}>, %arg1: !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
func.func @add_eint(%arg0: !FHE.eint<7>, %arg1: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.add_glwe"(%arg0, %arg1) {MANP = 2 : ui3} : (!TFHE.glwe<{_,_,_}{7}>, !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<{_,_,_}{7}>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<7>, !FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}

// RUN: concretecompiler %s --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func.func @neg_eint(%arg0: !TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
func.func @neg_eint(%arg0: !FHE.eint<7>) -> !FHE.eint<7> {
  // CHECK-NEXT:  %0 = "TFHE.neg_glwe"(%arg0) {MANP = 1 : ui1} : (!TFHE.glwe<{_,_,_}{7}>) -> !TFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT:  return %0 : !TFHE.glwe<{_,_,_}{7}>

  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<7>) -> (!FHE.eint<7>)
  return %1: !FHE.eint<7>
}

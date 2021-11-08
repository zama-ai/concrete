// RUN: zamacompiler %s --passes hlfhe-to-midlfhe --action=dump-midlfhe 2>&1| FileCheck %s

// CHECK-LABEL: func @neg_eint(%arg0: !MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}>
func @neg_eint(%arg0: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.neg_glwe"(%arg0) : (!MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{_,_,_}{7}>

  %1 = "HLFHE.neg_eint"(%arg0): (!HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}

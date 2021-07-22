// RUN: zamacompiler %s --passes hlfhe-to-midlfhe 2>&1| FileCheck %s

// CHECK-LABEL: func @add_eint(%arg0: !MidLFHE.glwe<{_,_,_}{7}>, %arg1: !MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}>
func @add_eint(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.add_glwe"(%arg0, %arg1) : (!MidLFHE.glwe<{_,_,_}{7}>, !MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{_,_,_}{7}>

  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
// RUN: zamacompiler %s --convert-hlfhe-to-midlfhe 2>&1| FileCheck %s

// CHECK-LABEL: func @mul_eint_int(%arg0: !MidLFHE.glwe<{_,_,_}{7}>) -> !MidLFHE.glwe<{_,_,_}{7}>
func @mul_eint_int(%arg0: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.mul_glwe_int"(%arg0, %[[V1]]) : (!MidLFHE.glwe<{_,_,_}{7}>, i8) -> !MidLFHE.glwe<{_,_,_}{7}>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.glwe<{_,_,_}{7}>

  %0 = constant 1 : i8
  %1 = "HLFHE.mul_eint_int"(%arg0, %0): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
// RUN: zamacompiler %s  2>&1| FileCheck %s


// CHECK-LABEL: func @add_plain_glwe(%arg0: !MidLFHE.glwe<1024,12>, %arg1: !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12>
func @add_plain_glwe(%arg0: !MidLFHE.glwe<1024,12>, %arg1: !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.h_mul"(%arg0, %arg1) : (!MidLFHE.glwe<1024,12>, !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<1024,12>

  %0 = "MidLFHE.h_mul"(%arg0, %arg1): (!MidLFHE.glwe<1024,12>, !MidLFHE.glwe<1024,12>) -> (!MidLFHE.glwe<1024,12>)
  return %0: !MidLFHE.glwe<1024,12>
}

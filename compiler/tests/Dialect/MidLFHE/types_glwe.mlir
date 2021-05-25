// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @glwe(%arg0: !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12>
func @glwe(%arg0: !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.glwe<1024,12>
  return %arg0: !MidLFHE.glwe<1024,12>
}
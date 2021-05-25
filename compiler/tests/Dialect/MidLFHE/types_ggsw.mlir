// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @ggsw(%arg0: !MidLFHE.ggsw<1024,12,3,2>) -> !MidLFHE.ggsw<1024,12,3,2>
func @ggsw(%arg0: !MidLFHE.ggsw<1024,12,3,2>) -> !MidLFHE.ggsw<1024,12,3,2> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.ggsw<1024,12,3,2>
  return %arg0: !MidLFHE.ggsw<1024,12,3,2>
}
// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @lwe(%arg0: !MidLFHE.lwe<1024>) -> !MidLFHE.lwe<1024>
func @lwe(%arg0: !MidLFHE.lwe<1024>) -> !MidLFHE.lwe<1024> {
  // CHECK-LABEL: return %arg0 : !MidLFHE.lwe<1024>
  return %arg0: !MidLFHE.lwe<1024>
}
// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @ciphertext(%arg0: !MidLFHE.ciphertext) -> !MidLFHE.ciphertext
func @ciphertext(%arg0: !MidLFHE.ciphertext) -> !MidLFHE.ciphertext {
  // CHECK-LABEL: return %arg0 : !MidLFHE.ciphertext
  return %arg0: !MidLFHE.ciphertext
}
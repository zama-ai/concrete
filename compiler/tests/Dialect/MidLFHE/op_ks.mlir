// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @keyswitch(%arg0: !MidLFHE.ksk, %arg1: !MidLFHE.ciphertext) -> !MidLFHE.ciphertext {
func @keyswitch(%arg0: !MidLFHE.ksk, %arg1: !MidLFHE.ciphertext) -> !MidLFHE.ciphertext {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.keyswitch"(%arg0, %arg1) {base_log = 8 : i32, level = 2 : i32} : (!MidLFHE.ksk, !MidLFHE.ciphertext) -> !MidLFHE.ciphertext
  %0 = "MidLFHE.keyswitch"(%arg0, %arg1) {base_log = 8 : i32, level = 2 : i32} : (!MidLFHE.ksk, !MidLFHE.ciphertext) -> !MidLFHE.ciphertext
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.ciphertext
  return %0 : !MidLFHE.ciphertext
}
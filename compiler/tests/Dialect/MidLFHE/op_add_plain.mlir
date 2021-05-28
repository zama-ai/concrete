// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @add_plain_ciphertext(%arg0: !MidLFHE.ciphertext) -> !MidLFHE.ciphertext
func @add_plain_ciphertext(%arg0: !MidLFHE.ciphertext) -> !MidLFHE.ciphertext {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.add_plain"(%arg0, %[[V1]]) : (!MidLFHE.ciphertext, i32) -> !MidLFHE.ciphertext
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.ciphertext

  %0 = constant 1 : i32
  %1 = "MidLFHE.add_plain"(%arg0, %0): (!MidLFHE.ciphertext, i32) -> (!MidLFHE.ciphertext)
  return %1: !MidLFHE.ciphertext
}

// CHECK-LABEL: func @add_plain_lwe(%arg0: !MidLFHE.lwe<1024>) -> !MidLFHE.lwe<1024>
func @add_plain_lwe(%arg0: !MidLFHE.lwe<1024>) -> !MidLFHE.lwe<1024> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.add_plain"(%arg0, %[[V1]]) : (!MidLFHE.lwe<1024>, i32) -> !MidLFHE.lwe<1024>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.lwe<1024>

  %0 = constant 1 : i32
  %1 = "MidLFHE.add_plain"(%arg0, %0): (!MidLFHE.lwe<1024>, i32) -> (!MidLFHE.lwe<1024>)
  return %1: !MidLFHE.lwe<1024>
}

// CHECK-LABEL: func @add_plain_glwe(%arg0: !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12>
func @add_plain_glwe(%arg0: !MidLFHE.glwe<1024,12>) -> !MidLFHE.glwe<1024,12> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.add_plain"(%arg0, %[[V1]]) : (!MidLFHE.glwe<1024,12>, i32) -> !MidLFHE.glwe<1024,12>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.glwe<1024,12>

  %0 = constant 1 : i32
  %1 = "MidLFHE.add_plain"(%arg0, %0): (!MidLFHE.glwe<1024,12>, i32) -> (!MidLFHE.glwe<1024,12>)
  return %1: !MidLFHE.glwe<1024,12>
}

// CHECK-LABEL: func @add_plain_ggsw(%arg0: !MidLFHE.ggsw<1024,12,3,2>) -> !MidLFHE.ggsw<1024,12,3,2>
func @add_plain_ggsw(%arg0: !MidLFHE.ggsw<1024,12,3,2>) -> !MidLFHE.ggsw<1024,12,3,2> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.add_plain"(%arg0, %[[V1]]) : (!MidLFHE.ggsw<1024,12,3,2>, i32) -> !MidLFHE.ggsw<1024,12,3,2>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.ggsw<1024,12,3,2>

  %0 = constant 1 : i32
  %1 = "MidLFHE.add_plain"(%arg0, %0): (!MidLFHE.ggsw<1024,12,3,2>, i32) -> (!MidLFHE.ggsw<1024,12,3,2>)
  return %1: !MidLFHE.ggsw<1024,12,3,2>
}
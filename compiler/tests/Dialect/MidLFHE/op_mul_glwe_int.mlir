// RUN: zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @mul_glwe_int(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
func @mul_glwe_int(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.mul_glwe_int"(%arg0, %[[V1]]) : (!MidLFHE.glwe<{1024,12,64}{7}>, i8) -> !MidLFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.glwe<{1024,12,64}{7}>

  %0 = arith.constant 1 : i8
  %1 = "MidLFHE.mul_glwe_int"(%arg0, %0): (!MidLFHE.glwe<{1024,12,64}{7}>, i8) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

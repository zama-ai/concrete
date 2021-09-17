// RUN: zamacompiler --entry-dialect=midlfhe --action=roundtrip %s  2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe_int(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
func @add_glwe_int(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.add_glwe_int"(%arg0, %[[V1]]) : (!MidLFHE.glwe<{1024,12,64}{7}>, i8) -> !MidLFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.glwe<{1024,12,64}{7}>

  %0 = constant 1 : i8
  %1 = "MidLFHE.add_glwe_int"(%arg0, %0): (!MidLFHE.glwe<{1024,12,64}{7}>, i8) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

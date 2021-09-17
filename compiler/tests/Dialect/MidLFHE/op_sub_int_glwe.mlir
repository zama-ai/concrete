// RUN: zamacompiler --entry-dialect=midlfhe --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @sub_int_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
func @sub_int_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "MidLFHE.sub_int_glwe"(%[[V1]], %arg0) : (i8, !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V2]] : !MidLFHE.glwe<{1024,12,64}{7}>

  %0 = constant 1 : i8
  %1 = "MidLFHE.sub_int_glwe"(%0, %arg0): (i8, !MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

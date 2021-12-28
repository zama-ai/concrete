// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.add_glwe"(%arg0, %arg1) : (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{1024,12,64}{7}>

  %0 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %0: !MidLFHE.glwe<{1024,12,64}{7}>
}

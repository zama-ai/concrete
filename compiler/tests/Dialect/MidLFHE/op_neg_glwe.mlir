// RUN: zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.neg_glwe"(%arg0) : (!MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{1024,12,64}{7}>

  %1 = "MidLFHE.neg_glwe"(%arg0): (!MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,12,64}{7}>
}

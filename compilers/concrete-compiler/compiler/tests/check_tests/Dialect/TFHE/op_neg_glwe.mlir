// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @neg_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}>
func.func @neg_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.neg_glwe"(%arg0) : (!TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<{1024,12,64}{7}>

  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<{1024,12,64}{7}>) -> (!TFHE.glwe<{1024,12,64}{7}>)
  return %1: !TFHE.glwe<{1024,12,64}{7}>
}

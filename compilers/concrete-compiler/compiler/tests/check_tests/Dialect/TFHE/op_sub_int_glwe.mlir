// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @sub_int_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}>
func.func @sub_int_glwe(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[V2:.*]] = "TFHE.sub_int_glwe"(%[[V1]], %arg0) : (i64, !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V2]] : !TFHE.glwe<{1024,12,64}{7}>

  %0 = arith.constant 1 : i64
  %1 = "TFHE.sub_int_glwe"(%0, %arg0): (i64, !TFHE.glwe<{1024,12,64}{7}>) -> (!TFHE.glwe<{1024,12,64}{7}>)
  return %1: !TFHE.glwe<{1024,12,64}{7}>
}

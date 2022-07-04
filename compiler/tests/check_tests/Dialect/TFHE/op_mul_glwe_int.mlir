// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @mul_glwe_int(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}>
func.func @mul_glwe_int(%arg0: !TFHE.glwe<{1024,12,64}{7}>) -> !TFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "TFHE.mul_glwe_int"(%arg0, %[[V1]]) : (!TFHE.glwe<{1024,12,64}{7}>, i8) -> !TFHE.glwe<{1024,12,64}{7}>
  // CHECK-NEXT: return %[[V2]] : !TFHE.glwe<{1024,12,64}{7}>

  %0 = arith.constant 1 : i8
  %1 = "TFHE.mul_glwe_int"(%arg0, %0): (!TFHE.glwe<{1024,12,64}{7}>, i8) -> (!TFHE.glwe<{1024,12,64}{7}>)
  return %1: !TFHE.glwe<{1024,12,64}{7}>
}

// RUN: concretecompiler --passes midlfhe-to-lowlfhe --action=dump-lowlfhe %s 2>&1| FileCheck %s

// CHECK-LABEL: func @mul_glwe_const_int(%arg0: !LowLFHE.lwe_ciphertext<1024,7>) -> !LowLFHE.lwe_ciphertext<1024,7>
func @mul_glwe_const_int(%arg0: !MidLFHE.glwe<{1024,1,64}{7}>) -> !MidLFHE.glwe<{1024,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "LowLFHE.int_to_cleartext"(%[[V1]]) : (i8) -> !LowLFHE.cleartext<8>
  // CHECK-NEXT: %[[V3:.*]] = "LowLFHE.mul_cleartext_lwe_ciphertext"(%arg0, %[[V2]]) : (!LowLFHE.lwe_ciphertext<1024,7>, !LowLFHE.cleartext<8>) -> !LowLFHE.lwe_ciphertext<1024,7>
  // CHECK-NEXT: return %[[V3]] : !LowLFHE.lwe_ciphertext<1024,7>
  %0 = arith.constant 1 : i8
  %1 = "MidLFHE.mul_glwe_int"(%arg0, %0): (!MidLFHE.glwe<{1024,1,64}{7}>, i8) -> (!MidLFHE.glwe<{1024,1,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,1,64}{7}>
}


// CHECK-LABEL: func @mul_glwe_int(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: i5) -> !LowLFHE.lwe_ciphertext<1024,4>
func @mul_glwe_int(%arg0: !MidLFHE.glwe<{1024,1,64}{4}>, %arg1: i5) -> !MidLFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.int_to_cleartext"(%arg1) : (i5) -> !LowLFHE.cleartext<5>
  // CHECK-NEXT: %[[V2:.*]] = "LowLFHE.mul_cleartext_lwe_ciphertext"(%arg0, %[[V1]]) : (!LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.cleartext<5>) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V2]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "MidLFHE.mul_glwe_int"(%arg0, %arg1): (!MidLFHE.glwe<{1024,1,64}{4}>, i5) -> (!MidLFHE.glwe<{1024,1,64}{4}>)
  return %1: !MidLFHE.glwe<{1024,1,64}{4}>
}

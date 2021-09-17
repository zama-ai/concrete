// RUN: zamacompiler --entry-dialect=midlfhe --action=dump-lowlfhe --parametrize-midlfhe=false %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe_const_int(%arg0: !LowLFHE.lwe_ciphertext<1024,7>) -> !LowLFHE.lwe_ciphertext<1024,7>
func @add_glwe_const_int(%arg0: !MidLFHE.glwe<{1024,1,64}{7}>) -> !MidLFHE.glwe<{1024,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "LowLFHE.encode_int"(%[[V1]]) : (i8) -> !LowLFHE.plaintext<8>
  // CHECK-NEXT: %[[V3:.*]] = "LowLFHE.add_plaintext_lwe_ciphertext"(%arg0, %[[V2]]) : (!LowLFHE.lwe_ciphertext<1024,7>, !LowLFHE.plaintext<8>) -> !LowLFHE.lwe_ciphertext<1024,7>
  // CHECK-NEXT: return %[[V3]] : !LowLFHE.lwe_ciphertext<1024,7>
  %0 = constant 1 : i8
  %1 = "MidLFHE.add_glwe_int"(%arg0, %0): (!MidLFHE.glwe<{1024,1,64}{7}>, i8) -> (!MidLFHE.glwe<{1024,1,64}{7}>)
  return %1: !MidLFHE.glwe<{1024,1,64}{7}>
}


// CHECK-LABEL: func @add_glwe_int(%arg0: !LowLFHE.lwe_ciphertext<1024,4>, %arg1: i5) -> !LowLFHE.lwe_ciphertext<1024,4>
func @add_glwe_int(%arg0: !MidLFHE.glwe<{1024,1,64}{4}>, %arg1: i5) -> !MidLFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.encode_int"(%arg1) : (i5) -> !LowLFHE.plaintext<5>
  // CHECK-NEXT: %[[V2:.*]] = "LowLFHE.add_plaintext_lwe_ciphertext"(%arg0, %[[V1]]) : (!LowLFHE.lwe_ciphertext<1024,4>, !LowLFHE.plaintext<5>) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V2]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "MidLFHE.add_glwe_int"(%arg0, %arg1): (!MidLFHE.glwe<{1024,1,64}{4}>, i5) -> (!MidLFHE.glwe<{1024,1,64}{4}>)
  return %1: !MidLFHE.glwe<{1024,1,64}{4}>
}

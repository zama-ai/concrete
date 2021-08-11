// RUN: zamacompiler --passes midlfhe-to-lowlfhe %s  2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe(%arg0: !LowLFHE.lwe_ciphertext, %arg1: !LowLFHE.lwe_ciphertext) -> !LowLFHE.lwe_ciphertext
func @add_glwe(%arg0: !MidLFHE.glwe<{1024,12,64}{7}>, %arg1: !MidLFHE.glwe<{1024,12,64}{7}>) -> !MidLFHE.glwe<{1024,12,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.add_lwe_ciphertexts"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext, !LowLFHE.lwe_ciphertext) -> !LowLFHE.lwe_ciphertext
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext

  %0 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{1024,12,64}{7}>, !MidLFHE.glwe<{1024,12,64}{7}>) -> (!MidLFHE.glwe<{1024,12,64}{7}>)
  return %0: !MidLFHE.glwe<{1024,12,64}{7}>
}

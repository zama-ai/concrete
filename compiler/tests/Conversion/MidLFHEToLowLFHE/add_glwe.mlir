// RUN: zamacompiler --entry-dialect=midlfhe --action=dump-lowlfhe --parametrize-midlfhe=false %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
func @add_glwe(%arg0: !MidLFHE.glwe<{2048,1,64}{7}>, %arg1: !MidLFHE.glwe<{2048,1,64}{7}>) -> !MidLFHE.glwe<{2048,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.add_lwe_ciphertexts"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %0 = "MidLFHE.add_glwe"(%arg0, %arg1): (!MidLFHE.glwe<{2048,1,64}{7}>, !MidLFHE.glwe<{2048,1,64}{7}>) -> (!MidLFHE.glwe<{2048,1,64}{7}>)
  return %0: !MidLFHE.glwe<{2048,1,64}{7}>
}

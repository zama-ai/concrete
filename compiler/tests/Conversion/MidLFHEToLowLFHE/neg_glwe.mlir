// RUN: concretecompiler --passes midlfhe-to-lowlfhe --action=dump-lowlfhe %s 2>&1| FileCheck %s

// CHECK-LABEL: func @neg_glwe(%arg0: !LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4>
func @neg_glwe(%arg0: !MidLFHE.glwe<{1024,1,64}{4}>) -> !MidLFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.negate_lwe_ciphertext"(%arg0) : (!LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<1024,4>
  %1 = "MidLFHE.neg_glwe"(%arg0): (!MidLFHE.glwe<{1024,1,64}{4}>) -> (!MidLFHE.glwe<{1024,1,64}{4}>)
  return %1: !MidLFHE.glwe<{1024,1,64}{4}>
}

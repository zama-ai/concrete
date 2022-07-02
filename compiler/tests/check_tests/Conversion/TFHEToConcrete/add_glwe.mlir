// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func @add_glwe(%arg0: !TFHE.glwe<{2048,1,64}{7}>, %arg1: !TFHE.glwe<{2048,1,64}{7}>) -> !TFHE.glwe<{2048,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %0 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<{2048,1,64}{7}>, !TFHE.glwe<{2048,1,64}{7}>) -> (!TFHE.glwe<{2048,1,64}{7}>)
  return %0: !TFHE.glwe<{2048,1,64}{7}>
}

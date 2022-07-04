// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @neg_glwe(%arg0: !Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
func.func @neg_glwe(%arg0: !TFHE.glwe<{1024,1,64}{4}>) -> !TFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<1024,4>
  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<{1024,1,64}{4}>) -> (!TFHE.glwe<{1024,1,64}{4}>)
  return %1: !TFHE.glwe<{1024,1,64}{4}>
}

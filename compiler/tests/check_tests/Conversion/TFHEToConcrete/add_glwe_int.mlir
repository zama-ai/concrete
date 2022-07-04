// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_glwe_const_int(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7>
func.func @add_glwe_const_int(%arg0: !TFHE.glwe<{1024,1,64}{7}>) -> !TFHE.glwe<{1024,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.encode_int"(%[[V1]]) : (i8) -> !Concrete.plaintext<8>
  // CHECK-NEXT: %[[V3:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %[[V2]]) : (!Concrete.lwe_ciphertext<1024,7>, !Concrete.plaintext<8>) -> !Concrete.lwe_ciphertext<1024,7>
  // CHECK-NEXT: return %[[V3]] : !Concrete.lwe_ciphertext<1024,7>
  %0 = arith.constant 1 : i8
  %1 = "TFHE.add_glwe_int"(%arg0, %0): (!TFHE.glwe<{1024,1,64}{7}>, i8) -> (!TFHE.glwe<{1024,1,64}{7}>)
  return %1: !TFHE.glwe<{1024,1,64}{7}>
}


// CHECK-LABEL: func.func @add_glwe_int(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4>
func.func @add_glwe_int(%arg0: !TFHE.glwe<{1024,1,64}{4}>, %arg1: i5) -> !TFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %[[V1]]) : (!Concrete.lwe_ciphertext<1024,4>, !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V2]] : !Concrete.lwe_ciphertext<1024,4>
  %1 = "TFHE.add_glwe_int"(%arg0, %arg1): (!TFHE.glwe<{1024,1,64}{4}>, i5) -> (!TFHE.glwe<{1024,1,64}{4}>)
  return %1: !TFHE.glwe<{1024,1,64}{4}>
}

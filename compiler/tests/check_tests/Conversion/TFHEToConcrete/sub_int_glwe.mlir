// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @sub_const_int_glwe(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7>
func.func @sub_const_int_glwe(%arg0: !TFHE.glwe<{1024,1,64}{7}>) -> !TFHE.glwe<{1024,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i8
  // CHECK-NEXT: %[[NEG:.*]] = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7>
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.encode_int"(%[[V1]]) : (i8) -> !Concrete.plaintext<8>
  // CHECK-NEXT: %[[V3:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%[[NEG]], %[[V2]]) : (!Concrete.lwe_ciphertext<1024,7>, !Concrete.plaintext<8>) -> !Concrete.lwe_ciphertext<1024,7>
  // CHECK-NEXT: return %[[V3]] : !Concrete.lwe_ciphertext<1024,7>
  %0 = arith.constant 1 : i8
  %1 = "TFHE.sub_int_glwe"(%0, %arg0): (i8, !TFHE.glwe<{1024,1,64}{7}>) -> (!TFHE.glwe<{1024,1,64}{7}>)
  return %1: !TFHE.glwe<{1024,1,64}{7}>
}

// CHECK-LABEL: func.func @sub_int_glwe(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4>
func.func @sub_int_glwe(%arg0: !TFHE.glwe<{1024,1,64}{4}>, %arg1: i5) -> !TFHE.glwe<{1024,1,64}{4}> {
  // CHECK-NEXT: %[[NEG:.*]] = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%[[NEG]], %[[V1]]) : (!Concrete.lwe_ciphertext<1024,4>, !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<1024,4>
  // CHECK-NEXT: return %[[V2]] : !Concrete.lwe_ciphertext<1024,4>
  %1 = "TFHE.sub_int_glwe"(%arg1, %arg0): (i5, !TFHE.glwe<{1024,1,64}{4}>) -> (!TFHE.glwe<{1024,1,64}{4}>)
  return %1: !TFHE.glwe<{1024,1,64}{4}>
}

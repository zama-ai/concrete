// RUN: concretecompiler --passes tfhe-optimization --action=dump-tfhe  --split-input-file --skip-program-info %s 2>&1| FileCheck %s


// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext(%arg0: !TFHE.glwe<sk[1]<527,1>>, %arg1: i64) -> !TFHE.glwe<sk[1]<527,1>>
func.func @mul_cleartext_lwe_ciphertext(%arg0: !TFHE.glwe<sk[1]<527,1>>, %arg1: i64) -> !TFHE.glwe<sk[1]<527,1>> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.mul_glwe_int"(%arg0, %arg1) : (!TFHE.glwe<sk[1]<527,1>>, i64) -> !TFHE.glwe<sk[1]<527,1>>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<sk[1]<527,1>>

  %1 = "TFHE.mul_glwe_int"(%arg0, %arg1): (!TFHE.glwe<sk[1]<527,1>>, i64) -> (!TFHE.glwe<sk[1]<527,1>>)
  return %1: !TFHE.glwe<sk[1]<527,1>>
}

// -----

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>>
func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.zero"() : () -> !TFHE.glwe<sk[1]<527,1>>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<sk[1]<527,1>>

  %0 = arith.constant 0 : i64
  %2 = "TFHE.mul_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<527,1>>, i64) -> (!TFHE.glwe<sk[1]<527,1>>)
  return %2: !TFHE.glwe<sk[1]<527,1>>
}

// -----

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext_minus_1(%arg0: !TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>>
func.func @mul_cleartext_lwe_ciphertext_minus_1(%arg0: !TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.neg_glwe"(%arg0) : (!TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<sk[1]<527,1>>

  %0 = arith.constant -1 : i64
  %2 = "TFHE.mul_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<527,1>>, i64) -> (!TFHE.glwe<sk[1]<527,1>>)
  return %2: !TFHE.glwe<sk[1]<527,1>>
}

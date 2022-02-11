// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s
  
// CHECK-LABEL: func @mul_lwe_const_int(%arg0: tensor<1025xi64>, %arg1: !Concrete.context) -> tensor<1025xi64>
func @mul_lwe_const_int(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
  // CHECK-NEXT: %c1_i8 = arith.constant 1 : i8
  // CHECK-NEXT: %0 = arith.extui %c1_i8 : i8 to i64
  // CHECK-NEXT: %1 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %2 = tensor.cast %1 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %3 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_mul_cleartext_lwe_ciphertext_u64(%2, %3, %0) : (tensor<?xi64>, tensor<?xi64>, i64) -> ()
  // CHECK-NEXT: return %1 : tensor<1025xi64>
  %c1_i8 = arith.constant 1 : i8
  %1 = "Concrete.int_to_cleartext"(%c1_i8) : (i8) -> !Concrete.cleartext<8>
  %2 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.mul_cleartext_lwe_buffer"(%2, %arg0, %1) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.cleartext<8>) -> ()
  return %2 : tensor<1025xi64>
}



// CHECK-LABEL: func @mul_lwe_int(%arg0: tensor<1025xi64>, %arg1: i5, %arg2: !Concrete.context) -> tensor<1025xi64>
func @mul_lwe_int(%arg0: tensor<1025xi64>, %arg1: i5) -> tensor<1025xi64> {
  // CHECK-NEXT: %0 = arith.extui %arg1 : i5 to i64
  // CHECK-NEXT: %1 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %2 = tensor.cast %1 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %3 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_mul_cleartext_lwe_ciphertext_u64(%2, %3, %0) : (tensor<?xi64>, tensor<?xi64>, i64) -> ()
  // CHECK-NEXT: return %1 : tensor<1025xi64>
  %0 = "Concrete.int_to_cleartext"(%arg1) : (i5) -> !Concrete.cleartext<5>
  %1 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.mul_cleartext_lwe_buffer"(%1, %arg0, %0) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.cleartext<5>) -> ()
  return %1 : tensor<1025xi64>
}

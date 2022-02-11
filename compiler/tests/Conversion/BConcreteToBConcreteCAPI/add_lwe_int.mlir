// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s


// CHECK-LABEL: func @add_glwe_const_int(%arg0: tensor<1025xi64>, %arg1: !Concrete.context) -> tensor<1025xi64>
func @add_glwe_const_int(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
  // CHECK-NEXT: %c1_i8 = arith.constant 1 : i8
  // CHECK-NEXT: %0 = arith.extui %c1_i8 : i8 to i64
  // CHECK-NEXT: %c56_i64 = arith.constant 56 : i64
  // CHECK-NEXT: %1 = arith.shli %0, %c56_i64 : i64
  // CHECK-NEXT: %2 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %3 = tensor.cast %2 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %4 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_add_plaintext_lwe_ciphertext_u64(%3, %4, %1) : (tensor<?xi64>, tensor<?xi64>, i64) -> ()
  // CHECK-NEXT: return %2 : tensor<1025xi64>
  %0 = arith.constant 1 : i8
  %1 = "Concrete.encode_int"(%0) : (i8) -> !Concrete.plaintext<8>
  %2 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.add_plaintext_lwe_buffer"(%2, %arg0, %1) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.plaintext<8>) -> ()
  return %2 : tensor<1025xi64>
}


// CHECK-LABEL: func @add_glwe_int(%arg0: tensor<1025xi64>, %arg1: i5, %arg2: !Concrete.context) -> tensor<1025xi64>
func @add_glwe_int(%arg0: tensor<1025xi64>, %arg1: i5) -> tensor<1025xi64> {
  // CHECK-NEXT: %0 = arith.extui %arg1 : i5 to i64
  // CHECK-NEXT: %c59_i64 = arith.constant 59 : i64
  // CHECK-NEXT: %1 = arith.shli %0, %c59_i64 : i64
  // CHECK-NEXT: %2 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %3 = tensor.cast %2 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %4 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_add_plaintext_lwe_ciphertext_u64(%3, %4, %1) : (tensor<?xi64>, tensor<?xi64>, i64) -> ()
  // CHECK-NEXT: return %2 : tensor<1025xi64>
  %0 = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  %1 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.add_plaintext_lwe_buffer"(%1, %arg0, %0) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.plaintext<5>) -> ()
  return %1 : tensor<1025xi64>
}

// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK-LABEL: func @sub_const_int_lwe(%arg0: tensor<1025xi64>, %arg1: !Concrete.context) -> tensor<1025xi64> {
func @sub_const_int_lwe(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
  // CHECK-NEXT: %c1_i8 = arith.constant 1 : i8
  // CHECK-NEXT: %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %1 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %2 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_negate_lwe_ciphertext_u64(%1, %2) : (tensor<?xi64>, tensor<?xi64>) -> ()
  // CHECK-NEXT: %3 = arith.extui %c1_i8 : i8 to i64
  // CHECK-NEXT: %c56_i64 = arith.constant 56 : i64
  // CHECK-NEXT: %4 = arith.shli %3, %c56_i64 : i64
  // CHECK-NEXT: %5 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %6 = tensor.cast %5 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %7 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_add_plaintext_lwe_ciphertext_u64(%6, %7, %4) : (tensor<?xi64>, tensor<?xi64>, i64) -> ()
  // CHECK-NEXT: return %5 : tensor<1025xi64>
  %0 = arith.constant 1 : i8
  %1 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.negate_lwe_buffer"(%1, %arg0) : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  %2 = "Concrete.encode_int"(%0) : (i8) -> !Concrete.plaintext<8>
  %3 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.add_plaintext_lwe_buffer"(%3, %1, %2) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.plaintext<8>) -> ()
  return %3 : tensor<1025xi64>
}

// CHECK-LABEL: func @sub_int_lwe(%arg0: tensor<1025xi64>, %arg1: i5, %arg2: !Concrete.context) -> tensor<1025xi64> { 
func @sub_int_lwe(%arg0: tensor<1025xi64>, %arg1: i5) -> tensor<1025xi64> {
  // CHECK-NEXT: %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %1 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %2 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_negate_lwe_ciphertext_u64(%1, %2) : (tensor<?xi64>, tensor<?xi64>) -> ()
  // CHECK-NEXT: %3 = arith.extui %arg1 : i5 to i64
  // CHECK-NEXT: %c59_i64 = arith.constant 59 : i64
  // CHECK-NEXT: %4 = arith.shli %3, %c59_i64 : i64
  // CHECK-NEXT: %5 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %6 = tensor.cast %5 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %7 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_add_plaintext_lwe_ciphertext_u64(%6, %7, %4) : (tensor<?xi64>, tensor<?xi64>, i64) -> ()
  // CHECK-NEXT: return %5 : tensor<1025xi64>
  %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.negate_lwe_buffer"(%0, %arg0) : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  %1 = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  %2 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.add_plaintext_lwe_buffer"(%2, %0, %1) : (tensor<1025xi64>, tensor<1025xi64>, !Concrete.plaintext<5>) -> ()
  return %2 : tensor<1025xi64>
}

// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK-LABEL: func @neg_lwe(%arg0: tensor<1025xi64>, %arg1: !Concrete.context) -> tensor<1025xi64> {
func @neg_lwe(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
  // CHECK-NEXT: %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: %1 = tensor.cast %0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: %2 = tensor.cast %arg0 : tensor<1025xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_negate_lwe_ciphertext_u64(%1, %2) : (tensor<?xi64>, tensor<?xi64>) -> ()
  // CHECK-NEXT: return %0 : tensor<1025xi64>
  %0 = linalg.init_tensor [1025] : tensor<1025xi64>
  "BConcrete.negate_lwe_buffer"(%0, %arg0) : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  return %0 : tensor<1025xi64>
}

// RUN: concretecompiler --passes bconcrete-to-bconcrete-c-api --action=dump-std %s 2>&1| FileCheck %s

// CHECK: func @add_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>, %arg2: !Concrete.context) -> tensor<2049xi64>
func @add_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64> {
  // CHECK-NEXT: %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: %1 = tensor.cast %0 : tensor<2049xi64> to tensor<?xi64>
  // CHECK-NEXT: %2 = tensor.cast %arg0 : tensor<2049xi64> to tensor<?xi64>
  // CHECK-NEXT: %3 = tensor.cast %arg1 : tensor<2049xi64> to tensor<?xi64>
  // CHECK-NEXT: call @memref_add_lwe_ciphertexts_u64(%1, %2, %3) : (tensor<?xi64>, tensor<?xi64>, tensor<?xi64>) -> ()
  // CHECK-NEXT: return %0 : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.add_lwe_buffer"(%0, %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, tensor<2049xi64>) -> ()
  return %0 : tensor<2049xi64>
}

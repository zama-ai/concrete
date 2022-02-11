// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_glwe(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64>
func @add_glwe(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.add_lwe_buffer"(%[[V1]], %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, tensor<2049xi64>) -> ()
  // CHECK-NEXT: return %[[V1]] : tensor<2049xi64>
  %0 = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  return %0 : !Concrete.lwe_ciphertext<2048,7>
}

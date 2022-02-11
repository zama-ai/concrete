// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func @neg_lwe(%arg0: tensor<1025xi64>) -> tensor<1025xi64>
func @neg_lwe(%arg0: !Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: %[[V1:.*]] = linalg.init_tensor [1025] : tensor<1025xi64>
  // CHECK-NEXT: "BConcrete.negate_lwe_buffer"(%[[V1]], %arg0) : (tensor<1025xi64>, tensor<1025xi64>) -> ()
  // CHECK-NEXT: return %[[V1]] : tensor<1025xi64>
  %0 = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
  return %0 : !Concrete.lwe_ciphertext<1024,4>
}

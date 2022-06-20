// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func @add_lwe_ciphertexts(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_lwe_buffer"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @add_lwe_ciphertexts(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  %0 = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  return %0 : !Concrete.lwe_ciphertext<2048,7>
}

//CHECK: func @add_crt_lwe_ciphertexts(%[[A0:.*]]: tensor<5x2049xi64>, %[[A1:.*]]: tensor<5x2049xi64>) -> tensor<5x2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_crt_lwe_buffer"(%[[A0]], %[[A1]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, tensor<5x2049xi64>) -> tensor<5x2049xi64>
//CHECK:   return %[[V0]] : tensor<5x2049xi64>
//CHECK: }
func.func @add_crt_lwe_ciphertexts(%arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>, %arg1: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7> {
  %0 = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>, !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>
  return %0 : !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>
}

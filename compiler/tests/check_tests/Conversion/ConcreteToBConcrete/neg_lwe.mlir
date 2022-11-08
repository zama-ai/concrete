// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @neg_lwe(%[[A0:.*]]: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.negate_lwe_tensor"(%[[A0]]) : (tensor<1025xi64>) -> tensor<1025xi64>
//CHECK:   return %[[V0]] : tensor<1025xi64>
//CHECK: }
func.func @neg_lwe(%arg0: !Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4> {
  %0 = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
  return %0 : !Concrete.lwe_ciphertext<1024,4>
}

//CHECK: func.func @negate_lwe_ciphertext(%[[A0:.*]]: tensor<5x1025xi64>) -> tensor<5x1025xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.negate_crt_lwe_tensor"(%[[A0]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x1025xi64>) -> tensor<5x1025xi64>
//CHECK:   return %[[V0]] : tensor<5x1025xi64>
//CHECK: }
func.func @negate_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4> {
  %0 = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>
  return %0 : !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>
}

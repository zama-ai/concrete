// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @mul_lwe_const_int(%[[A0:.*]]: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %c1_i8 = arith.constant 1 : i8
//CHECK:   %[[V0:.*]] = arith.extui %c1_i8 : i8 to i64
//CHECK:   %[[V1:.*]] = "BConcrete.mul_cleartext_lwe_buffer"(%[[A0]], %[[V0]]) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %[[V1]] : tensor<1025xi64>
//CHECK: }
func.func @mul_lwe_const_int(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
  %0 = arith.constant 1 : i8
  %2 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %0) : (!Concrete.lwe_ciphertext<1024,7>, i8) -> !Concrete.lwe_ciphertext<1024,7>
  return %2 : !Concrete.lwe_ciphertext<1024,7>
}

//CHECK: func.func @mul_lwe_int(%[[A0:.*]]: tensor<1025xi64>, %[[A1:.*]]: i5) -> tensor<1025xi64> {
//CHECK:   %[[V0:.*]] = arith.extui %[[A1]] : i5 to i64
//CHECK:   %[[V1:.*]] = "BConcrete.mul_cleartext_lwe_buffer"(%[[A0]], %[[V0]]) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %[[V1]] : tensor<1025xi64>
//CHECK: }
func.func @mul_lwe_int(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4> {
  %1 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<1024,4>, i5) -> !Concrete.lwe_ciphertext<1024,4>
  return %1 : !Concrete.lwe_ciphertext<1024,4>
}


//CHECK: func.func @mul_cleartext_lwe_ciphertext_crt(%[[A0:.*]]: tensor<5x1025xi64>, %[[A1:.*]]: i5) -> tensor<5x1025xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.mul_cleartext_crt_lwe_buffer"(%[[A0]], %[[A1]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x1025xi64>, i5) -> tensor<5x1025xi64>
//CHECK:   return %[[V0]] : tensor<5x1025xi64>
//CHECK: }
func.func @mul_cleartext_lwe_ciphertext_crt(%arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4> {
  %1 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>, i5) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>
  return %1 : !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,4>
}

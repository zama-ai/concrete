// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s


//CHECK: func.func @add_glwe_const_int(%[[A0:.*]]: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %c1_i8 = arith.constant 1 : i8
//CHECK:   %[[V0:.*]] = arith.extui %c1_i8 : i8 to i64
//CHECK:   %c56_i64 = arith.constant 56 : i64
//CHECK:   %[[V1:.*]] = arith.shli %[[V0]], %c56_i64 : i64
//CHECK:   %[[V2:.*]] = "BConcrete.add_plaintext_lwe_buffer"(%[[A0]], %[[V1]]) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %[[V2]] : tensor<1025xi64>
//CHECK: }
func.func @add_glwe_const_int(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
  %0 = arith.constant 1 : i8
  %2 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %0) : (!Concrete.lwe_ciphertext<1024,7>, i8) -> !Concrete.lwe_ciphertext<1024,7>
  return %2 : !Concrete.lwe_ciphertext<1024,7>
}

//CHECK: func.func @add_glwe_int(%[[A0:.*]]: tensor<1025xi64>, %[[A1:.*]]: i5) -> tensor<1025xi64> {
//CHECK:   %[[V0:.*]] = arith.extui %[[A1]] : i5 to i64
//CHECK:   %c59_i64 = arith.constant 59 : i64
//CHECK:   %[[V1:.*]] = arith.shli %[[V0]], %c59_i64 : i64
//CHECK:   %[[V2:.*]] = "BConcrete.add_plaintext_lwe_buffer"(%[[A0]], %[[V1]]) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %[[V2]] : tensor<1025xi64>
//CHECK: }
func.func @add_glwe_int(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4> {
  %1 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<1024,4>, i5) -> !Concrete.lwe_ciphertext<1024,4>
  return %1 : !Concrete.lwe_ciphertext<1024,4>
}


//CHECK: func.func @add_plaintext_lwe_ciphertext(%[[A0:.*]]: tensor<5x1025xi64>) -> tensor<5x1025xi64> {
//CHECK:   %c1_i8 = arith.constant 1 : i8
//CHECK:   %[[V0:.*]] = "BConcrete.add_plaintext_crt_lwe_buffer"(%[[A0]], %c1_i8) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x1025xi64>, i8) -> tensor<5x1025xi64>
//CHECK:   return %[[V0]] : tensor<5x1025xi64>
//CHECK: }
func.func @add_plaintext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7> {
  %0 = arith.constant 1 : i8
  %2 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %0) : (!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>, i8) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>
  return %2 : !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>
}

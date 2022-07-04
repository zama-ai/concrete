// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @mul_lwe_const_int(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %c1_i8 = arith.constant 1 : i8
//CHECK:   %0 = arith.extui %c1_i8 : i8 to i64
//CHECK:   %1 = "BConcrete.mul_cleartext_lwe_buffer"(%arg0, %0) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %1 : tensor<1025xi64>
//CHECK: }

func.func @mul_lwe_const_int(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
  %0 = arith.constant 1 : i8
  %1 = "Concrete.int_to_cleartext"(%0) : (i8) -> !Concrete.cleartext<8>
  %2 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %1) : (!Concrete.lwe_ciphertext<1024,7>, !Concrete.cleartext<8>) -> !Concrete.lwe_ciphertext<1024,7>
  return %2 : !Concrete.lwe_ciphertext<1024,7>
}

//CHECK: func.func @mul_lwe_int(%arg0: tensor<1025xi64>, %arg1: i5) -> tensor<1025xi64> {
//CHECK:   %0 = arith.extui %arg1 : i5 to i64
//CHECK:   %1 = "BConcrete.mul_cleartext_lwe_buffer"(%arg0, %0) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %1 : tensor<1025xi64>
//CHECK: }
func.func @mul_lwe_int(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4> {
  %0 = "Concrete.int_to_cleartext"(%arg1) : (i5) -> !Concrete.cleartext<5>
  %1 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %0) : (!Concrete.lwe_ciphertext<1024,4>, !Concrete.cleartext<5>) -> !Concrete.lwe_ciphertext<1024,4>
  return %1 : !Concrete.lwe_ciphertext<1024,4>
}

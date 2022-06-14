// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s


//CHECK: func @add_glwe_const_int(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %c1_i8 = arith.constant 1 : i8
//CHECK:   %0 = arith.extui %c1_i8 : i8 to i64
//CHECK:   %c56_i64 = arith.constant 56 : i64
//CHECK:   %1 = arith.shli %0, %c56_i64 : i64
//CHECK:   %2 = "BConcrete.add_plaintext_lwe_buffer"(%arg0, %1) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %2 : tensor<1025xi64>
//CHECK: }
func @add_glwe_const_int(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
  %0 = arith.constant 1 : i8
  %1 = "Concrete.encode_int"(%0) : (i8) -> !Concrete.plaintext<8>
  %2 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %1) : (!Concrete.lwe_ciphertext<1024,7>, !Concrete.plaintext<8>) -> !Concrete.lwe_ciphertext<1024,7>
  return %2 : !Concrete.lwe_ciphertext<1024,7>
}

//CHECK: func @add_glwe_int(%arg0: tensor<1025xi64>, %arg1: i5) -> tensor<1025xi64> {
//CHECK:   %0 = arith.extui %arg1 : i5 to i64
//CHECK:   %c59_i64 = arith.constant 59 : i64
//CHECK:   %1 = arith.shli %0, %c59_i64 : i64
//CHECK:   %2 = "BConcrete.add_plaintext_lwe_buffer"(%arg0, %1) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %2 : tensor<1025xi64>
//CHECK: }
func @add_glwe_int(%arg0: !Concrete.lwe_ciphertext<1024,4>, %arg1: i5) -> !Concrete.lwe_ciphertext<1024,4> {
  %0 = "Concrete.encode_int"(%arg1) : (i5) -> !Concrete.plaintext<5>
  %1 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %0) : (!Concrete.lwe_ciphertext<1024,4>, !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<1024,4>
  return %1 : !Concrete.lwe_ciphertext<1024,4>
}

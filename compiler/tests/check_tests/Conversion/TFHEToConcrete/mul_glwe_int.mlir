// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

//CHECK: func.func @mul_glwe_const_int(%[[A0:.*]]: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
//CHECK:   %c1_i64 = arith.constant 1 : i64
//CHECK:   %[[V0:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%[[A0]], %c1_i64) : (!Concrete.lwe_ciphertext<1024,7>, i64) -> !Concrete.lwe_ciphertext<1024,7>
//CHECK:   return %[[V0]] : !Concrete.lwe_ciphertext<1024,7>
//CHECK: }
func.func @mul_glwe_const_int(%arg0: !TFHE.glwe<{1024,1,64}{7}>) -> !TFHE.glwe<{1024,1,64}{7}> {
  %0 = arith.constant 1 : i64
  %1 = "TFHE.mul_glwe_int"(%arg0, %0): (!TFHE.glwe<{1024,1,64}{7}>, i64) -> (!TFHE.glwe<{1024,1,64}{7}>)
  return %1: !TFHE.glwe<{1024,1,64}{7}>
}


//CHECK: func.func @mul_glwe_int(%[[A0:.*]]: !Concrete.lwe_ciphertext<1024,4>, %[[A1:.*]]: i64) -> !Concrete.lwe_ciphertext<1024,4> {
//CHECK:   %[[V0:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%[[A0]], %[[A1]]) : (!Concrete.lwe_ciphertext<1024,4>, i64) -> !Concrete.lwe_ciphertext<1024,4>
//CHECK:   return %[[V0]] : !Concrete.lwe_ciphertext<1024,4>
//CHECK: }
func.func @mul_glwe_int(%arg0: !TFHE.glwe<{1024,1,64}{4}>, %arg1: i64) -> !TFHE.glwe<{1024,1,64}{4}> {
  %1 = "TFHE.mul_glwe_int"(%arg0, %arg1): (!TFHE.glwe<{1024,1,64}{4}>, i64) -> (!TFHE.glwe<{1024,1,64}{4}>)
  return %1: !TFHE.glwe<{1024,1,64}{4}>
}

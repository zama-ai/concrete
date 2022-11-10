// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

//CHECK: func.func @sub_const_int_glwe(%[[A0:.*]]: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
//CHECK:   %c1_i64 = arith.constant 1 : i64
//CHECK:   %[[V0:.*]] = "Concrete.negate_lwe_ciphertext"(%[[A0]]) : (!Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7>
//CHECK:   %[[V1:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%[[V0]], %c1_i64) : (!Concrete.lwe_ciphertext<1024,7>, i64) -> !Concrete.lwe_ciphertext<1024,7>
//CHECK:   return %[[V1]] : !Concrete.lwe_ciphertext<1024,7>
//CHECK: }
func.func @sub_const_int_glwe(%arg0: !TFHE.glwe<{1024,1,64}{7}>) -> !TFHE.glwe<{1024,1,64}{7}> {
  %0 = arith.constant 1 : i64
  %1 = "TFHE.sub_int_glwe"(%0, %arg0): (i64, !TFHE.glwe<{1024,1,64}{7}>) -> (!TFHE.glwe<{1024,1,64}{7}>)
  return %1: !TFHE.glwe<{1024,1,64}{7}>
}

//CHECK: func.func @sub_int_glwe(%[[A0:.*]]: !Concrete.lwe_ciphertext<1024,4>, %[[A1:.*]]: i64) -> !Concrete.lwe_ciphertext<1024,4> {
//CHECK:   %[[V0:.*]] = "Concrete.negate_lwe_ciphertext"(%[[A0]]) : (!Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<1024,4>
//CHECK:   %[[V1:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%[[V0]], %[[A1]]) : (!Concrete.lwe_ciphertext<1024,4>, i64) -> !Concrete.lwe_ciphertext<1024,4>
//CHECK:   return %[[V1]] : !Concrete.lwe_ciphertext<1024,4>
//CHECK: }
func.func @sub_int_glwe(%arg0: !TFHE.glwe<{1024,1,64}{4}>, %arg1: i64) -> !TFHE.glwe<{1024,1,64}{4}> {
  %1 = "TFHE.sub_int_glwe"(%arg1, %arg0): (i64, !TFHE.glwe<{1024,1,64}{4}>) -> (!TFHE.glwe<{1024,1,64}{4}>)
  return %1: !TFHE.glwe<{1024,1,64}{4}>
}

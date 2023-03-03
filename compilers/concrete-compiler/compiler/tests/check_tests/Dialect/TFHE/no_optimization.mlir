// RUN: concretecompiler --passes tfhe-optimization --optimize-tfhe=false --action=dump-tfhe %s 2>&1| FileCheck %s

//CHECK: func.func @mul_cleartext_glwe_ciphertext_0(%[[A0:.*]]: !TFHE.glwe<{1,527,64}{7}>) -> !TFHE.glwe<{1,527,64}{7}> {
//CHECK:   %c0_i64 = arith.constant 0 : i64
//CHECK:   %[[V0:.*]] = "TFHE.mul_glwe_int"(%[[A0]], %c0_i64) : (!TFHE.glwe<{1,527,64}{7}>, i64) -> !TFHE.glwe<{1,527,64}{7}>
//CHECK:   return %[[V0]] : !TFHE.glwe<{1,527,64}{7}>
//CHECK: }
func.func @mul_cleartext_glwe_ciphertext_0(%arg0: !TFHE.glwe<{1,527,64}{7}>) -> !TFHE.glwe<{1,527,64}{7}> {
  %0 = arith.constant 0 : i64
  %2 = "TFHE.mul_glwe_int"(%arg0, %0): (!TFHE.glwe<{1,527,64}{7}>, i64) -> (!TFHE.glwe<{1,527,64}{7}>)
  return %2: !TFHE.glwe<{1,527,64}{7}>
}

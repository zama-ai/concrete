// RUN: concretecompiler --passes tfhe-optimization --optimize-tfhe=false --action=dump-tfhe --skip-program-info %s 2>&1| FileCheck %s

//CHECK: func.func @mul_cleartext_glwe_ciphertext_0(%[[A0:.*]]: !TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>> {
//CHECK:   %c0_i64 = arith.constant 0 : i64
//CHECK:   %[[V0:.*]] = "TFHE.mul_glwe_int"(%[[A0]], %c0_i64) : (!TFHE.glwe<sk[1]<527,1>>, i64) -> !TFHE.glwe<sk[1]<527,1>>
//CHECK:   return %[[V0]] : !TFHE.glwe<sk[1]<527,1>>
//CHECK: }
func.func @mul_cleartext_glwe_ciphertext_0(%arg0: !TFHE.glwe<sk[1]<527,1>>) -> !TFHE.glwe<sk[1]<527,1>> {
  %0 = arith.constant 0 : i64
  %2 = "TFHE.mul_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<527,1>>, i64) -> (!TFHE.glwe<sk[1]<527,1>>)
  return %2: !TFHE.glwe<sk[1]<527,1>>
}

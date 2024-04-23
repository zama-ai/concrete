// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete --split-input-file --skip-program-info %s 2>&1| FileCheck %s


//CHECK: func.func @add_glwe_const_int(%[[A0:.*]]: tensor<1025xi64>) -> tensor<1025xi64> {
//CHECK:   %c1_i64 = arith.constant 1 : i64
//CHECK:   %[[V0:.*]] = "Concrete.add_plaintext_lwe_tensor"(%[[A0]], %c1_i64) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %[[V0]] : tensor<1025xi64>
//CHECK: }
func.func @add_glwe_const_int(%arg0: !TFHE.glwe<sk[1]<1,1024>>) -> !TFHE.glwe<sk[1]<1,1024>> {
  %0 = arith.constant 1 : i64
  %1 = "TFHE.add_glwe_int"(%arg0, %0): (!TFHE.glwe<sk[1]<1,1024>>, i64) -> (!TFHE.glwe<sk[1]<1,1024>>)
  return %1: !TFHE.glwe<sk[1]<1,1024>>
}

// -----

//CHECK: func.func @add_glwe_int(%[[A0:.*]]: tensor<1025xi64>, %[[A1:.*]]: i64) -> tensor<1025xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.add_plaintext_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<1025xi64>, i64) -> tensor<1025xi64>
//CHECK:   return %[[V0]] : tensor<1025xi64>
//CHECK: }
func.func @add_glwe_int(%arg0: !TFHE.glwe<sk[1]<1,1024>>, %arg1: i64) -> !TFHE.glwe<sk[1]<1,1024>> {
  %1 = "TFHE.add_glwe_int"(%arg0, %arg1): (!TFHE.glwe<sk[1]<1,1024>>, i64) -> (!TFHE.glwe<sk[1]<1,1024>>)
  return %1: !TFHE.glwe<sk[1]<1,1024>>
}

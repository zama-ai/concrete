// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @apply_lookup_table_cst(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
//CHECK:   %[[V0:.*]] = bufferization.alloc_tensor() : tensor<4096xi64>
//CHECK:   "BConcrete.fill_glwe_from_table"(%[[V0]], %cst) {glweDimension = 1 : i32, outPrecision = 4 : i32, polynomialSize = 2048 : i32} : (tensor<4096xi64>, tensor<16xi64>) -> ()
//CHECK:   %[[V1:.*]] = "BConcrete.keyswitch_lwe_buffer"(%[[A0]]) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 600 : i32} : (tensor<2049xi64>) -> tensor<601xi64>
//CHECK:   %[[V2:.*]] = "BConcrete.bootstrap_lwe_buffer"(%[[V1]], %[[V0]]) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 2048 : i32} : (tensor<601xi64>, tensor<4096xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V2]] : tensor<2049xi64>
//CHECK: }
func.func @apply_lookup_table_cst(%arg0: !Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<2048,4> {
  %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  %0 = "Concrete.glwe_from_table"(%tlu) {glweDimension = 1 : i32, p = 4 : i32, polynomialSize = 2048 : i32} : (tensor<16xi64>) -> !Concrete.glwe_ciphertext<1, 2048,4>
  %1 = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 600 : i32} : (!Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<600,4>
  %2 = "Concrete.bootstrap_lwe"(%1, %0) {baseLog = 4 : i32, glweDimension = 1 : i32, level = 5 : i32, polynomialSize = 2048 : i32} : (!Concrete.lwe_ciphertext<600,4>, !Concrete.glwe_ciphertext<1,2048,4>) -> !Concrete.lwe_ciphertext<2048,4>
  return %2 : !Concrete.lwe_ciphertext<2048,4>
}

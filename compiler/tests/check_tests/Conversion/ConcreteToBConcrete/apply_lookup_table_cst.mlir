// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @apply_lookup_table_cst(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
//CHECK:   %[[V1:.*]] = "BConcrete.keyswitch_lwe_tensor"(%[[A0]]) {baseLog = 2 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 600 : i32} : (tensor<2049xi64>) -> tensor<601xi64>
//CHECK:   %[[V2:.*]] = "BConcrete.bootstrap_lwe_tensor"(%[[V1]], %cst) {baseLog = 2 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (tensor<601xi64>, tensor<16xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V2]] : tensor<2049xi64>
//CHECK: }
func.func @apply_lookup_table_cst(%arg0: !Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<2048,4> {
  %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  %1 = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<600,4>
  %2 = "Concrete.bootstrap_lwe"(%1, %tlu) {baseLog = 2 : i32, polySize = 2048 : i32, level = 3 : i32, glweDimension = 4 : i32} : (!Concrete.lwe_ciphertext<600,4>, tensor<16xi64>) -> !Concrete.lwe_ciphertext<2048,4>
  return %2 : !Concrete.lwe_ciphertext<2048,4>
}

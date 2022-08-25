// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

//CHECK: func.func @apply_lookup_table_cst(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[C1:.*]] = arith.constant 600 : i32
//CHECK:   %[[C2:.*]] = arith.constant 2048 : i32
//CHECK:   %[[C3:.*]] = arith.constant 4 : i32
//CHECK:   %[[C4:.*]] = arith.constant 5 : i32
//CHECK:   %[[C5:.*]] = arith.constant 1 : i32
//CHECK:   %cst = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
//CHECK:   %[[V1:.*]] = "BConcrete.keyswitch_lwe_buffer"(%[[A0]]) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 600 : i32} : (tensor<2049xi64>) -> tensor<601xi64>
//CHECK:   %[[V2:.*]] = "BConcrete.bootstrap_lwe_buffer"(%[[V1]], %cst, %[[C1]], %[[C2]], %[[C3]], %[[C4]], %[[C5]], %[[C3]]) : (tensor<601xi64>, tensor<16xi64>, i32, i32, i32, i32, i32, i32) -> tensor<2049xi64>
//CHECK:   return %[[V2]] : tensor<2049xi64>
//CHECK: }
func.func @apply_lookup_table_cst(%arg0: !Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<2048,4> {
  %c1 = arith.constant 600 : i32
  %c2 = arith.constant 2048 : i32
  %c3 = arith.constant 4 : i32
  %c4 = arith.constant 5 : i32
  %c5 = arith.constant 1 : i32
  %c6 = arith.constant 4 : i32
  %tlu = arith.constant dense<[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]> : tensor<16xi64>
  %1 = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 600 : i32} : (!Concrete.lwe_ciphertext<2048,4>) -> !Concrete.lwe_ciphertext<600,4>
  %2 = "Concrete.bootstrap_lwe"(%1, %tlu, %c1, %c2, %c3, %c4, %c5, %c6) : (!Concrete.lwe_ciphertext<600,4>, tensor<16xi64>, i32, i32, i32, i32, i32, i32) -> !Concrete.lwe_ciphertext<2048,4>
  return %2 : !Concrete.lwe_ciphertext<2048,4>
}

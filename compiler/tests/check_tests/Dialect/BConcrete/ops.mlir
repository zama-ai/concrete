// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

//CHECK: func @add_lwe_ciphertexts(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_lwe_buffer"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func @add_lwe_ciphertexts(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.add_lwe_buffer"(%arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>) -> ( tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func @add_plaintext_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: i64) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_plaintext_lwe_buffer"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, i64) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func @add_plaintext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: i64) -> tensor<2049xi64> {
  %0 = "BConcrete.add_plaintext_lwe_buffer"(%arg0, %arg1) : (tensor<2049xi64>, i64) -> ( tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func @mul_cleartext_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: i64) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.mul_cleartext_lwe_buffer"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, i64) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func @mul_cleartext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: i64) -> tensor<2049xi64> {
  %0 = "BConcrete.mul_cleartext_lwe_buffer"(%arg0, %arg1) : (tensor<2049xi64>, i64) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func @negate_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.negate_lwe_buffer"(%[[A0]]) : (tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func @negate_lwe_ciphertext(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.negate_lwe_buffer"(%arg0) : (tensor<2049xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func @bootstrap_lwe(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<4096xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.bootstrap_lwe_buffer"(%[[A0]], %[[A1]]) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 1024 : i32} : (tensor<2049xi64>, tensor<4096xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<4096xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.bootstrap_lwe_buffer"(%arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 1024 : i32} : (tensor<2049xi64>, tensor<4096xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func @keyswitch_lwe(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.keyswitch_lwe_buffer"(%[[A0]]) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 1 : i32} : (tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func @keyswitch_lwe(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.keyswitch_lwe_buffer"(%arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 1 : i32} : (tensor<2049xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

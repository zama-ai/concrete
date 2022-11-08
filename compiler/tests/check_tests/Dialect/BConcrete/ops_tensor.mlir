// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

//CHECK: func.func @add_lwe_ciphertexts(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @add_lwe_ciphertexts(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.add_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>) -> ( tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @add_crt_lwe_ciphertexts(%[[A0:.*]]: tensor<5x2049xi64>, %[[A1:.*]]: tensor<5x2049xi64>) -> tensor<5x2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_crt_lwe_tensor"(%[[A0]], %[[A1]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, tensor<5x2049xi64>) -> tensor<5x2049xi64>
//CHECK:   return %[[V0]] : tensor<5x2049xi64>
//CHECK: }
func.func @add_crt_lwe_ciphertexts(%arg0: tensor<5x2049xi64>, %arg1: tensor<5x2049xi64>) -> tensor<5x2049xi64> {
  %0 = "BConcrete.add_crt_lwe_tensor"(%arg0, %arg1) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, tensor<5x2049xi64>) -> ( tensor<5x2049xi64>)
  return %0 : tensor<5x2049xi64>
}

//CHECK: func.func @add_plaintext_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: i64) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_plaintext_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, i64) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @add_plaintext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: i64) -> tensor<2049xi64> {
  %0 = "BConcrete.add_plaintext_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, i64) -> ( tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @add_plaintext_crt_lwe_ciphertext(%[[A0:.*]]: tensor<5x2049xi64>, %[[A1:.*]]: i64) -> tensor<5x2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.add_plaintext_crt_lwe_tensor"(%[[A0]], %[[A1]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, i64) -> tensor<5x2049xi64>
//CHECK:   return %[[V0]] : tensor<5x2049xi64>
//CHECK: }
func.func @add_plaintext_crt_lwe_ciphertext(%arg0: tensor<5x2049xi64>, %arg1: i64) -> tensor<5x2049xi64> {
  %0 = "BConcrete.add_plaintext_crt_lwe_tensor"(%arg0, %arg1) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, i64) -> ( tensor<5x2049xi64>)
  return %0 : tensor<5x2049xi64>
}

//CHECK: func @mul_cleartext_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: i64) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.mul_cleartext_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, i64) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @mul_cleartext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: i64) -> tensor<2049xi64> {
  %0 = "BConcrete.mul_cleartext_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, i64) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @mul_cleartext_crt_lwe_ciphertext(%[[A0:.*]]: tensor<5x2049xi64>, %[[A1:.*]]: i64) -> tensor<5x2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.mul_cleartext_crt_lwe_tensor"(%[[A0]], %[[A1]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, i64) -> tensor<5x2049xi64>
//CHECK:   return %[[V0]] : tensor<5x2049xi64>
//CHECK: }
func.func @mul_cleartext_crt_lwe_ciphertext(%arg0: tensor<5x2049xi64>, %arg1: i64) -> tensor<5x2049xi64> {
  %0 = "BConcrete.mul_cleartext_crt_lwe_tensor"(%arg0, %arg1) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>, i64) -> (tensor<5x2049xi64>)
  return %0 : tensor<5x2049xi64>
}

//CHECK: func.func @negate_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.negate_lwe_tensor"(%[[A0]]) : (tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @negate_lwe_ciphertext(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.negate_lwe_tensor"(%arg0) : (tensor<2049xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @negate_crt_lwe_ciphertext(%[[A0:.*]]: tensor<5x2049xi64>) -> tensor<5x2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.negate_crt_lwe_tensor"(%[[A0]]) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>) -> tensor<5x2049xi64>
//CHECK:   return %[[V0]] : tensor<5x2049xi64>
//CHECK: }
func.func @negate_crt_lwe_ciphertext(%arg0: tensor<5x2049xi64>) -> tensor<5x2049xi64> {
  %0 = "BConcrete.negate_crt_lwe_tensor"(%arg0) {crtDecomposition = [2, 3, 5, 7, 11]} : (tensor<5x2049xi64>) -> (tensor<5x2049xi64>)
  return %0 : tensor<5x2049xi64>
}

//CHECK: func.func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<16xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.bootstrap_lwe_tensor"(%arg0, %arg1) {baseLog = 2 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (tensor<2049xi64>, tensor<16xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<16xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.bootstrap_lwe_tensor"(%arg0, %arg1) {baseLog = 2 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (tensor<2049xi64>, tensor<16xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @keyswitch_lwe(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "BConcrete.keyswitch_lwe_tensor"(%[[A0]]) {baseLog = 2 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @keyswitch_lwe(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "BConcrete.keyswitch_lwe_tensor"(%arg0) {baseLog = 2 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (tensor<2049xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

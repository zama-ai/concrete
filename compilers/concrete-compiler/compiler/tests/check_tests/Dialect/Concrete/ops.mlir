// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// Tensor ops

//CHECK: func.func @add_lwe_ciphertexts(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.add_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @add_lwe_ciphertexts(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "Concrete.add_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>) -> ( tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @add_plaintext_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: i64) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.add_plaintext_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, i64) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @add_plaintext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: i64) -> tensor<2049xi64> {
  %0 = "Concrete.add_plaintext_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, i64) -> ( tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func @mul_cleartext_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>, %[[A1:.*]]: i64) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.mul_cleartext_lwe_tensor"(%[[A0]], %[[A1]]) : (tensor<2049xi64>, i64) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @mul_cleartext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: i64) -> tensor<2049xi64> {
  %0 = "Concrete.mul_cleartext_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, i64) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @negate_lwe_ciphertext(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.negate_lwe_tensor"(%[[A0]]) : (tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @negate_lwe_ciphertext(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "Concrete.negate_lwe_tensor"(%arg0) : (tensor<2049xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<16xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.bootstrap_lwe_tensor"(%arg0, %arg1) {baseLog = 2 : i32, bskIndex = 0 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (tensor<2049xi64>, tensor<16xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: tensor<16xi64>) -> tensor<2049xi64> {
  %0 = "Concrete.bootstrap_lwe_tensor"(%arg0, %arg1) {baseLog = 2 : i32, bskIndex = 0 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (tensor<2049xi64>, tensor<16xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

//CHECK: func.func @keyswitch_lwe(%[[A0:.*]]: tensor<2049xi64>) -> tensor<2049xi64> {
//CHECK:   %[[V0:.*]] = "Concrete.keyswitch_lwe_tensor"(%[[A0]]) {baseLog = 2 : i32, kskIndex = 0 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (tensor<2049xi64>) -> tensor<2049xi64>
//CHECK:   return %[[V0]] : tensor<2049xi64>
//CHECK: }
func.func @keyswitch_lwe(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  %0 = "Concrete.keyswitch_lwe_tensor"(%arg0) {baseLog = 2 : i32, kskIndex = 0 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (tensor<2049xi64>) -> (tensor<2049xi64>)
  return %0 : tensor<2049xi64>
}

// MemRef Ops

func.func @add_lwe_ciphertexts_buffer(%arg0: memref<2049xi64>, %arg1: memref<2049xi64>, %result : memref<2049xi64>) {
  //CHECK: "Concrete.add_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A1:.*]]) : (memref<2049xi64>, memref<2049xi64>, memref<2049xi64>) -> ()
  "Concrete.add_lwe_buffer"(%result, %arg0, %arg1) : (memref<2049xi64>, memref<2049xi64>, memref<2049xi64>) -> ()
  return
}

func.func @add_plaintext_lwe_ciphertext_buffer(%arg0: memref<2049xi64>, %arg1: i64, %result: memref<2049xi64>) {
  //CHECK: "Concrete.add_plaintext_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A1:.*]]) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  "Concrete.add_plaintext_lwe_buffer"(%result, %arg0, %arg1) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  return
}

func.func @mul_cleartext_lwe_ciphertext_buffer(%arg0: memref<2049xi64>, %arg1: i64, %result: memref<2049xi64>) {
  //CHECK: "Concrete.mul_cleartext_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A0:.*]]) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  "Concrete.mul_cleartext_lwe_buffer"(%result, %arg0, %arg1) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  return
}

func.func @negate_lwe_ciphertext_buffer(%arg0: memref<2049xi64>, %result: memref<2049xi64>) {
  //CHECK: "Concrete.negate_lwe_buffer"(%[[R:.*]], %[[A0:.*]]) : (memref<2049xi64>, memref<2049xi64>) -> ()
  "Concrete.negate_lwe_buffer"(%result, %arg0) : (memref<2049xi64>, memref<2049xi64>) -> ()
  return
}

func.func @bootstrap_lwe_buffer(%arg0: memref<2049xi64>, %arg1: memref<16xi64>, %result: memref<2049xi64>) {
  //CHECK: "Concrete.bootstrap_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A1:.*]]) {baseLog = 2 : i32, bskIndex = 0 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>, memref<16xi64>) -> ()
  "Concrete.bootstrap_lwe_buffer"(%result, %arg0, %arg1) {baseLog = 2 : i32, bskIndex = 0 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>, memref<16xi64>) -> ()
  return
}

func.func @keyswitch_lwe_buffer(%arg0: memref<2049xi64>, %result: memref<2049xi64>) {
  //CHECK: "Concrete.keyswitch_lwe_buffer"(%[[R:.*]], %[[A0:.*]]) {baseLog = 2 : i32, kskIndex = 0 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>) -> ()
  "Concrete.keyswitch_lwe_buffer"(%result, %arg0) {baseLog = 2 : i32, kskIndex = 0 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>) -> ()
  return
}

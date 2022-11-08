// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

func.func @add_lwe_ciphertexts(%arg0: memref<2049xi64>, %arg1: memref<2049xi64>, %result : memref<2049xi64>) {
  //CHECK: "BConcrete.add_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A1:.*]]) : (memref<2049xi64>, memref<2049xi64>, memref<2049xi64>) -> ()
  "BConcrete.add_lwe_buffer"(%result, %arg0, %arg1) : (memref<2049xi64>, memref<2049xi64>, memref<2049xi64>) -> ()
  return
}

func.func @add_plaintext_lwe_ciphertext(%arg0: memref<2049xi64>, %arg1: i64, %result: memref<2049xi64>) {
  //CHECK: "BConcrete.add_plaintext_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A1:.*]]) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  "BConcrete.add_plaintext_lwe_buffer"(%result, %arg0, %arg1) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  return
}

func.func @mul_cleartext_lwe_ciphertext(%arg0: memref<2049xi64>, %arg1: i64, %result: memref<2049xi64>) {
  //CHECK: "BConcrete.mul_cleartext_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A0:.*]]) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  "BConcrete.mul_cleartext_lwe_buffer"(%result, %arg0, %arg1) : (memref<2049xi64>, memref<2049xi64>, i64) -> ()
  return
}

func.func @negate_lwe_ciphertext(%arg0: memref<2049xi64>, %result: memref<2049xi64>) {
  //CHECK: "BConcrete.negate_lwe_buffer"(%[[R:.*]], %[[A0:.*]]) : (memref<2049xi64>, memref<2049xi64>) -> ()
  "BConcrete.negate_lwe_buffer"(%result, %arg0) : (memref<2049xi64>, memref<2049xi64>) -> ()
  return
}

func.func @bootstrap_lwe(%arg0: memref<2049xi64>, %arg1: memref<16xi64>, %result: memref<2049xi64>) {
  //CHECK: "BConcrete.bootstrap_lwe_buffer"(%[[R:.*]], %[[A0:.*]], %[[A1:.*]]) {baseLog = 2 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>, memref<16xi64>) -> ()
  "BConcrete.bootstrap_lwe_buffer"(%result, %arg0, %arg1) {baseLog = 2 : i32, glweDimension = 4 : i32, inputLweDim = 600 : i32, level = 3 : i32, outPrecision = 4 : i32, polySize = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>, memref<16xi64>) -> ()
  return
}

func.func @keyswitch_lwe(%arg0: memref<2049xi64>, %result: memref<2049xi64>) {
  //CHECK: "BConcrete.keyswitch_lwe_buffer"(%[[R:.*]], %[[A0:.*]]) {baseLog = 2 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>) -> ()
  "BConcrete.keyswitch_lwe_buffer"(%result, %arg0) {baseLog = 2 : i32, level = 3 : i32, lwe_dim_in = 2048 : i32, lwe_dim_out = 2048 : i32} : (memref<2049xi64>, memref<2049xi64>) -> ()
  return
}

// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_lwe_ciphertexts(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64>
func @add_lwe_ciphertexts(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64> {
  // CHECK-NEXT: %[[V0:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.add_lwe_buffer"(%[[V0]], %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, tensor<2049xi64>) -> ()
  // CHECK-NEXT: return %[[V0]] : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.add_lwe_buffer"(%0, %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, tensor<2049xi64>) -> ()
  return %0 : tensor<2049xi64>
}

// CHECK-LABEL: func @add_plaintext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: !Concrete.plaintext<5>) -> tensor<2049xi64>
func @add_plaintext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: !Concrete.plaintext<5>) -> tensor<2049xi64> {
  // CHECK-NEXT: %[[V0:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.add_plaintext_lwe_buffer"(%[[V0]], %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, !Concrete.plaintext<5>) -> ()
  // CHECK-NEXT: return %[[V0]] : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.add_plaintext_lwe_buffer"(%0, %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, !Concrete.plaintext<5>) -> ()
  return %0 : tensor<2049xi64>
}

// CHECK-LABEL: func @mul_cleartext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: !Concrete.cleartext<7>) -> tensor<2049xi64>
func @mul_cleartext_lwe_ciphertext(%arg0: tensor<2049xi64>, %arg1: !Concrete.cleartext<7>) -> tensor<2049xi64> {
  // CHECK-NEXT: %[[V0:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.mul_cleartext_lwe_buffer"(%[[V0]], %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, !Concrete.cleartext<7>) -> ()
  // CHECK-NEXT: return %[[V0]] : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.mul_cleartext_lwe_buffer"(%0, %arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>, !Concrete.cleartext<7>) -> ()
  return %0 : tensor<2049xi64>
}

// CHECK-LABEL: func @negate_lwe_ciphertext(%arg0: tensor<2049xi64>) -> tensor<2049xi64>
func @negate_lwe_ciphertext(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  // CHECK-NEXT: %[[V0:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.negate_lwe_buffer"(%[[V0]], %arg0) : (tensor<2049xi64>, tensor<2049xi64>) -> ()
  // CHECK-NEXT: return %[[V0]] : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.negate_lwe_buffer"(%0, %arg0) : (tensor<2049xi64>, tensor<2049xi64>) -> ()
  return %0 : tensor<2049xi64>
}

// CHECK-LABEL: func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: !Concrete.glwe_ciphertext) -> tensor<2049xi64>
func @bootstrap_lwe(%arg0: tensor<2049xi64>, %arg1: !Concrete.glwe_ciphertext) -> tensor<2049xi64> {
  // CHECK-NEXT: %[[V0:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.bootstrap_lwe_buffer"(%[[V0]], %arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 1024 : i32} : (tensor<2049xi64>, tensor<2049xi64>, !Concrete.glwe_ciphertext) -> ()
  // CHECK-NEXT: return %[[V0]] : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.bootstrap_lwe_buffer"(%0, %arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 1024 : i32} : (tensor<2049xi64>, tensor<2049xi64>, !Concrete.glwe_ciphertext) -> ()
  return %0 : tensor<2049xi64>
}

// CHECK-LABEL: func @keyswitch_lwe(%arg0: tensor<2049xi64>) -> tensor<2049xi64>
func @keyswitch_lwe(%arg0: tensor<2049xi64>) -> tensor<2049xi64> {
  // CHECK-NEXT: %[[V0:.*]] = linalg.init_tensor [2049] : tensor<2049xi64>
  // CHECK-NEXT: "BConcrete.keyswitch_lwe_buffer"(%[[V0]], %arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 1 : i32} : (tensor<2049xi64>, tensor<2049xi64>) -> ()
  // CHECK-NEXT: return %[[V0]] : tensor<2049xi64>
  %0 = linalg.init_tensor [2049] : tensor<2049xi64>
  "BConcrete.keyswitch_lwe_buffer"(%0, %arg0) {baseLog = 2 : i32, inputLweDimension = 1 : i32, level = 3 : i32, outputLweDimension = 1 : i32} : (tensor<2049xi64>, tensor<2049xi64>) -> ()
  return %0 : tensor<2049xi64>
}

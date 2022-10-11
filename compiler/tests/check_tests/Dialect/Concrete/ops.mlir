// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_lwe_ciphertexts(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @add_lwe_ciphertexts(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @add_plaintext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: i5) -> !Concrete.lwe_ciphertext<2048,7>
func.func @add_plaintext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: i5) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, i5) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, i5) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: i7) -> !Concrete.lwe_ciphertext<2048,7>
func.func @mul_cleartext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: i7) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, i7) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, i7) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @negate_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @negate_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.negate_lwe_ciphertext"(%arg0): (!Concrete.lwe_ciphertext<2048,7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @bootstrap_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: tensor<128xi64>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @bootstrap_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: tensor<128xi64>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.bootstrap_lwe"(%arg0, %arg1) {baseLog = 2 : i32, glweDimension = 4 : i32, level = 3 : i32, polySize = 2048 : i32} : (!Concrete.lwe_ciphertext<2048,7>, tensor<128xi64>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>
  %1 = "Concrete.bootstrap_lwe"(%arg0, %arg1) {baseLog = 2 : i32, polySize = 2048 : i32, level = 3 : i32, glweDimension = 4 : i32} : (!Concrete.lwe_ciphertext<2048,7>, tensor<128xi64>) -> !Concrete.lwe_ciphertext<2048,7>
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @keyswitch_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @keyswitch_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>
  %1 = "Concrete.keyswitch_lwe"(%arg0){baseLog = 2 : i32, level = 3 : i32}: (!Concrete.lwe_ciphertext<2048,7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

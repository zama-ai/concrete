// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_lwe_ciphertexts(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @add_lwe_ciphertexts(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.add_lwe_ciphertexts"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, !Concrete.lwe_ciphertext<2048,7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @add_plaintext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @add_plaintext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.plaintext<5>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.add_plaintext_lwe_ciphertext"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, !Concrete.plaintext<5>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.cleartext<7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @mul_cleartext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.cleartext<7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.cleartext<7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, !Concrete.cleartext<7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @negate_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @negate_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.negate_lwe_ciphertext"(%arg0): (!Concrete.lwe_ciphertext<2048,7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @bootstrap_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.glwe_ciphertext<2048,1,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @bootstrap_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: !Concrete.glwe_ciphertext<2048,1,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.bootstrap_lwe"(%arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 2048 : i32} : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.glwe_ciphertext<2048,1,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>
  %1 = "Concrete.bootstrap_lwe"(%arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 2048 : i32} : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.glwe_ciphertext<2048,1,7>) -> !Concrete.lwe_ciphertext<2048,7>
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @keyswitch_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @keyswitch_lwe(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>
  %1 = "Concrete.keyswitch_lwe"(%arg0){baseLog = 2 : i32, level = 3 : i32}: (!Concrete.lwe_ciphertext<2048,7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @encode_int(%arg0: i6) -> !Concrete.plaintext<6>
func.func @encode_int(%arg0: i6) -> (!Concrete.plaintext<6>) {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.encode_int"(%arg0) : (i6) -> !Concrete.plaintext<6>
  // CHECK-NEXT: return %[[V1]] : !Concrete.plaintext<6>

  %0 = "Concrete.encode_int"(%arg0): (i6) -> !Concrete.plaintext<6>
  return %0: !Concrete.plaintext<6>
}

// CHECK-LABEL: func.func @int_to_cleartext() -> !Concrete.cleartext<6>
func.func @int_to_cleartext() -> !Concrete.cleartext<6> {
  // CHECK-NEXT: %[[V0:.*]] = arith.constant 5 : i6
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.int_to_cleartext"(%[[V0]]) : (i6) -> !Concrete.cleartext<6>
  // CHECK-NEXT: return %[[V1]] : !Concrete.cleartext<6>
  %0 = arith.constant 5 : i6
  %1 = "Concrete.int_to_cleartext"(%0) : (i6) -> !Concrete.cleartext<6>
  return %1 : !Concrete.cleartext<6>
}

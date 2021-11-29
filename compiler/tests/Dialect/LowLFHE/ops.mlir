// RUN: zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @add_lwe_ciphertexts(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
func @add_lwe_ciphertexts(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.add_lwe_ciphertexts"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %1 = "LowLFHE.add_lwe_ciphertexts"(%arg0, %arg1): (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.lwe_ciphertext<2048,7>) -> (!LowLFHE.lwe_ciphertext<2048,7>)
  return %1: !LowLFHE.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @add_plaintext_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.plaintext<5>) -> !LowLFHE.lwe_ciphertext<2048,7>
func @add_plaintext_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.plaintext<5>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.add_plaintext_lwe_ciphertext"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.plaintext<5>) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %1 = "LowLFHE.add_plaintext_lwe_ciphertext"(%arg0, %arg1): (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.plaintext<5>) -> (!LowLFHE.lwe_ciphertext<2048,7>)
  return %1: !LowLFHE.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @mul_cleartext_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.cleartext<7>) -> !LowLFHE.lwe_ciphertext<2048,7>
func @mul_cleartext_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.cleartext<7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.mul_cleartext_lwe_ciphertext"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.cleartext<7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %1 = "LowLFHE.mul_cleartext_lwe_ciphertext"(%arg0, %arg1): (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.cleartext<7>) -> (!LowLFHE.lwe_ciphertext<2048,7>)
  return %1: !LowLFHE.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @negate_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
func @negate_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.negate_lwe_ciphertext"(%arg0) : (!LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %1 = "LowLFHE.negate_lwe_ciphertext"(%arg0): (!LowLFHE.lwe_ciphertext<2048,7>) -> (!LowLFHE.lwe_ciphertext<2048,7>)
  return %1: !LowLFHE.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @bootstrap_lwe(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<2048,7>
func @bootstrap_lwe(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.bootstrap_lwe"(%arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 1024 : i32} : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.glwe_ciphertext) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %1 = "LowLFHE.bootstrap_lwe"(%arg0, %arg1) {baseLog = -1 : i32, glweDimension = 1 : i32, level = -1 : i32, polynomialSize = 1024 : i32} : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.glwe_ciphertext) -> (!LowLFHE.lwe_ciphertext<2048,7>)
  return %1: !LowLFHE.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @keyswitch_lwe(%arg0: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
func @keyswitch_lwe(%arg0: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.keyswitch_lwe"(%arg0) {baseLog = 2 : i32, level = 3 : i32} : (!LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.lwe_ciphertext<2048,7>

  %1 = "LowLFHE.keyswitch_lwe"(%arg0){baseLog = 2 : i32, level = 3 : i32}: (!LowLFHE.lwe_ciphertext<2048,7>) -> (!LowLFHE.lwe_ciphertext<2048,7>)
  return %1: !LowLFHE.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @encode_int(%arg0: i6) -> !LowLFHE.plaintext<6>
func @encode_int(%arg0: i6) -> (!LowLFHE.plaintext<6>) {
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.encode_int"(%arg0) : (i6) -> !LowLFHE.plaintext<6>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.plaintext<6>

  %0 = "LowLFHE.encode_int"(%arg0): (i6) -> !LowLFHE.plaintext<6>
  return %0: !LowLFHE.plaintext<6>
}

// CHECK-LABEL: func @int_to_cleartext() -> !LowLFHE.cleartext<6>
func @int_to_cleartext() -> !LowLFHE.cleartext<6> {
  // CHECK-NEXT: %[[V0:.*]] = arith.constant 5 : i6
  // CHECK-NEXT: %[[V1:.*]] = "LowLFHE.int_to_cleartext"(%[[V0]]) : (i6) -> !LowLFHE.cleartext<6>
  // CHECK-NEXT: return %[[V1]] : !LowLFHE.cleartext<6>
  %0 = arith.constant 5 : i6
  %1 = "LowLFHE.int_to_cleartext"(%0) : (i6) -> !LowLFHE.cleartext<6>
  return %1 : !LowLFHE.cleartext<6>
}

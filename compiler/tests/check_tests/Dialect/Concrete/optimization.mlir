// RUN: concretecompiler --action=dump-concrete %s 2>&1| FileCheck %s


// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: i7) -> !Concrete.lwe_ciphertext<2048,7>
func.func @mul_cleartext_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>, %arg1: i7) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1) : (!Concrete.lwe_ciphertext<2048,7>, i7) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %1 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %arg1): (!Concrete.lwe_ciphertext<2048,7>, i7) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %1: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.zero"() : () -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %0 = arith.constant 0 : i7
  %2 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %0): (!Concrete.lwe_ciphertext<2048,7>, i7) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %2: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext_minus_1(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @mul_cleartext_lwe_ciphertext_minus_1(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.negate_lwe_ciphertext"(%arg0) : (!Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V1]] : !Concrete.lwe_ciphertext<2048,7>

  %0 = arith.constant -1 : i7
  %2 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %0): (!Concrete.lwe_ciphertext<2048,7>, i7) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %2: !Concrete.lwe_ciphertext<2048,7>
}

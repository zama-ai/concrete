// RUN: concretecompiler --optimize-concrete=false --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: %[[V0:.*]] = arith.constant 0 : i7
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.int_to_cleartext"(%[[V0]]) : (i7) -> !Concrete.cleartext<7>
  // CHECK-NEXT: %[[V2:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %[[V1]]) : (!Concrete.lwe_ciphertext<2048,7>, !Concrete.cleartext<7>) -> !Concrete.lwe_ciphertext<2048,7>
  // CHECK-NEXT: return %[[V2]] : !Concrete.lwe_ciphertext<2048,7>

  %0 = arith.constant 0 : i7
  %1 = "Concrete.int_to_cleartext"(%0) : (i7) -> !Concrete.cleartext<7>
  %2 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %1): (!Concrete.lwe_ciphertext<2048,7>, !Concrete.cleartext<7>) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %2: !Concrete.lwe_ciphertext<2048,7>
}

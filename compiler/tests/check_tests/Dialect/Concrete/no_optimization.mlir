// RUN: concretecompiler --optimize-concrete=false --action=dump-concrete %s 2>&1| FileCheck %s

//CHECK: func.func @mul_cleartext_lwe_ciphertext_0(%[[A0:.*]]: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
//CHECK:   %c0_i7 = arith.constant 0 : i7
//CHECK:   %[[V0:.*]] = "Concrete.mul_cleartext_lwe_ciphertext"(%[[A0]], %c0_i7) : (!Concrete.lwe_ciphertext<2048,7>, i7) -> !Concrete.lwe_ciphertext<2048,7>
//CHECK:   return %[[V0]] : !Concrete.lwe_ciphertext<2048,7>
//CHECK: }
func.func @mul_cleartext_lwe_ciphertext_0(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  %0 = arith.constant 0 : i7
  %2 = "Concrete.mul_cleartext_lwe_ciphertext"(%arg0, %0): (!Concrete.lwe_ciphertext<2048,7>, i7) -> (!Concrete.lwe_ciphertext<2048,7>)
  return %2: !Concrete.lwe_ciphertext<2048,7>
}

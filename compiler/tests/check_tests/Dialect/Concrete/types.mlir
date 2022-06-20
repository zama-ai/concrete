// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s


// CHECK-LABEL: func.func @type_plaintext(%arg0: !Concrete.plaintext<7>) -> !Concrete.plaintext<7>
func.func @type_plaintext(%arg0: !Concrete.plaintext<7>) -> !Concrete.plaintext<7> {
  // CHECK-NEXT: return %arg0 : !Concrete.plaintext<7>
  return %arg0: !Concrete.plaintext<7>
}

// CHECK-LABEL: func.func @type_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func.func @type_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: return %arg0 : !Concrete.lwe_ciphertext<2048,7>
  return %arg0: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @type_lwe_ciphertext_with_crt(%arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>
func.func @type_lwe_ciphertext_with_crt(%arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>) -> !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7> {
  // CHECK-NEXT: return %arg0 : !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>
  return %arg0: !Concrete.lwe_ciphertext<crt=[2,3,5,7,11],2048,7>
}

// CHECK-LABEL: func @type_cleartext(%arg0: !Concrete.cleartext<5>) -> !Concrete.cleartext<5>
func.func @type_cleartext(%arg0: !Concrete.cleartext<5>) -> !Concrete.cleartext<5> {
  // CHECK-NEXT: return %arg0 : !Concrete.cleartext<5>
  return %arg0: !Concrete.cleartext<5>
}

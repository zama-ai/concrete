// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s


// CHECK-LABEL: func @type_plaintext(%arg0: !Concrete.plaintext<7>) -> !Concrete.plaintext<7>
func @type_plaintext(%arg0: !Concrete.plaintext<7>) -> !Concrete.plaintext<7> {
  // CHECK-NEXT: return %arg0 : !Concrete.plaintext<7>
  return %arg0: !Concrete.plaintext<7>
}

// CHECK-LABEL: func @type_plaintext_list(%arg0: !Concrete.plaintext_list) -> !Concrete.plaintext_list
func @type_plaintext_list(%arg0: !Concrete.plaintext_list) -> !Concrete.plaintext_list {
  // CHECK-NEXT: return %arg0 : !Concrete.plaintext_list
  return %arg0: !Concrete.plaintext_list
}

// CHECK-LABEL: func @type_foreign_plaintext_list(%arg0: !Concrete.foreign_plaintext_list) -> !Concrete.foreign_plaintext_list
func @type_foreign_plaintext_list(%arg0: !Concrete.foreign_plaintext_list) -> !Concrete.foreign_plaintext_list {
  // CHECK-NEXT: return %arg0 : !Concrete.foreign_plaintext_list
  return %arg0: !Concrete.foreign_plaintext_list
}

// CHECK-LABEL: func @type_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7>
func @type_lwe_ciphertext(%arg0: !Concrete.lwe_ciphertext<2048,7>) -> !Concrete.lwe_ciphertext<2048,7> {
  // CHECK-NEXT: return %arg0 : !Concrete.lwe_ciphertext<2048,7>
  return %arg0: !Concrete.lwe_ciphertext<2048,7>
}

// CHECK-LABEL: func @type_lwe_key_switch_key(%arg0: !Concrete.lwe_key_switch_key) -> !Concrete.lwe_key_switch_key
func @type_lwe_key_switch_key(%arg0: !Concrete.lwe_key_switch_key) -> !Concrete.lwe_key_switch_key {
  // CHECK-NEXT: return %arg0 : !Concrete.lwe_key_switch_key
  return %arg0: !Concrete.lwe_key_switch_key
}

// CHECK-LABEL: func @type_lwe_bootstrap_key(%arg0: !Concrete.lwe_bootstrap_key) -> !Concrete.lwe_bootstrap_key
func @type_lwe_bootstrap_key(%arg0: !Concrete.lwe_bootstrap_key) -> !Concrete.lwe_bootstrap_key {
  // CHECK-NEXT: return %arg0 : !Concrete.lwe_bootstrap_key
  return %arg0: !Concrete.lwe_bootstrap_key
}

// CHECK-LABEL: func @type_cleartext(%arg0: !Concrete.cleartext<5>) -> !Concrete.cleartext<5>
func @type_cleartext(%arg0: !Concrete.cleartext<5>) -> !Concrete.cleartext<5> {
  // CHECK-NEXT: return %arg0 : !Concrete.cleartext<5>
  return %arg0: !Concrete.cleartext<5>
}

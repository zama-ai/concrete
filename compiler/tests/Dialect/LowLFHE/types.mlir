// RUN: zamacompiler --round-trip %s  2>&1| FileCheck %s


// CHECK-LABEL: func @type_enc_rand_gen(%arg0: !LowLFHE.enc_rand_gen) -> !LowLFHE.enc_rand_gen
func @type_enc_rand_gen(%arg0: !LowLFHE.enc_rand_gen) -> !LowLFHE.enc_rand_gen {
  // CHECK-NEXT: return %arg0 : !LowLFHE.enc_rand_gen
  return %arg0: !LowLFHE.enc_rand_gen
}

// CHECK-LABEL: func @type_secret_rand_gen(%arg0: !LowLFHE.secret_rand_gen) -> !LowLFHE.secret_rand_gen
func @type_secret_rand_gen(%arg0: !LowLFHE.secret_rand_gen) -> !LowLFHE.secret_rand_gen {
  // CHECK-NEXT: return %arg0 : !LowLFHE.secret_rand_gen
  return %arg0: !LowLFHE.secret_rand_gen
}

// CHECK-LABEL: func @type_plaintext(%arg0: !LowLFHE.plaintext) -> !LowLFHE.plaintext
func @type_plaintext(%arg0: !LowLFHE.plaintext) -> !LowLFHE.plaintext {
  // CHECK-NEXT: return %arg0 : !LowLFHE.plaintext
  return %arg0: !LowLFHE.plaintext
}

// CHECK-LABEL: func @type_plaintext_list(%arg0: !LowLFHE.plaintext_list) -> !LowLFHE.plaintext_list
func @type_plaintext_list(%arg0: !LowLFHE.plaintext_list) -> !LowLFHE.plaintext_list {
  // CHECK-NEXT: return %arg0 : !LowLFHE.plaintext_list
  return %arg0: !LowLFHE.plaintext_list
}

// CHECK-LABEL: func @type_foreign_plaintext_list(%arg0: !LowLFHE.foreign_plaintext_list) -> !LowLFHE.foreign_plaintext_list
func @type_foreign_plaintext_list(%arg0: !LowLFHE.foreign_plaintext_list) -> !LowLFHE.foreign_plaintext_list {
  // CHECK-NEXT: return %arg0 : !LowLFHE.foreign_plaintext_list
  return %arg0: !LowLFHE.foreign_plaintext_list
}

// CHECK-LABEL: func @type_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext) -> !LowLFHE.lwe_ciphertext
func @type_lwe_ciphertext(%arg0: !LowLFHE.lwe_ciphertext) -> !LowLFHE.lwe_ciphertext {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_ciphertext
  return %arg0: !LowLFHE.lwe_ciphertext
}

// CHECK-LABEL: func @type_lwe_key_switch_key(%arg0: !LowLFHE.lwe_key_switch_key) -> !LowLFHE.lwe_key_switch_key
func @type_lwe_key_switch_key(%arg0: !LowLFHE.lwe_key_switch_key) -> !LowLFHE.lwe_key_switch_key {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_key_switch_key
  return %arg0: !LowLFHE.lwe_key_switch_key
}

// CHECK-LABEL: func @type_lwe_bootstrap_key(%arg0: !LowLFHE.lwe_bootstrap_key) -> !LowLFHE.lwe_bootstrap_key
func @type_lwe_bootstrap_key(%arg0: !LowLFHE.lwe_bootstrap_key) -> !LowLFHE.lwe_bootstrap_key {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_bootstrap_key
  return %arg0: !LowLFHE.lwe_bootstrap_key
}

// CHECK-LABEL: func @type_lwe_secret_key(%arg0: !LowLFHE.lwe_secret_key) -> !LowLFHE.lwe_secret_key
func @type_lwe_secret_key(%arg0: !LowLFHE.lwe_secret_key) -> !LowLFHE.lwe_secret_key {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_secret_key
  return %arg0: !LowLFHE.lwe_secret_key
}

// CHECK-LABEL: func @type_lwe_size(%arg0: !LowLFHE.lwe_size) -> !LowLFHE.lwe_size
func @type_lwe_size(%arg0: !LowLFHE.lwe_size) -> !LowLFHE.lwe_size {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_size
  return %arg0: !LowLFHE.lwe_size
}

// CHECK-LABEL: func @type_glwe_ciphertext(%arg0: !LowLFHE.glwe_ciphertext) -> !LowLFHE.glwe_ciphertext
func @type_glwe_ciphertext(%arg0: !LowLFHE.glwe_ciphertext) -> !LowLFHE.glwe_ciphertext {
  // CHECK-NEXT: return %arg0 : !LowLFHE.glwe_ciphertext
  return %arg0: !LowLFHE.glwe_ciphertext
}

// CHECK-LABEL: func @type_glwe_secret_key(%arg0: !LowLFHE.glwe_secret_key) -> !LowLFHE.glwe_secret_key
func @type_glwe_secret_key(%arg0: !LowLFHE.glwe_secret_key) -> !LowLFHE.glwe_secret_key {
  // CHECK-NEXT: return %arg0 : !LowLFHE.glwe_secret_key
  return %arg0: !LowLFHE.glwe_secret_key
}

// CHECK-LABEL: func @type_glwe_size(%arg0: !LowLFHE.glwe_size) -> !LowLFHE.glwe_size
func @type_glwe_size(%arg0: !LowLFHE.glwe_size) -> !LowLFHE.glwe_size {
  // CHECK-NEXT: return %arg0 : !LowLFHE.glwe_size
  return %arg0: !LowLFHE.glwe_size
}

// CHECK-LABEL: func @type_polynomial_size(%arg0: !LowLFHE.polynomial_size) -> !LowLFHE.polynomial_size
func @type_polynomial_size(%arg0: !LowLFHE.polynomial_size) -> !LowLFHE.polynomial_size {
  // CHECK-NEXT: return %arg0 : !LowLFHE.polynomial_size
  return %arg0: !LowLFHE.polynomial_size
}

// CHECK-LABEL: func @type_decomp_level_count(%arg0: !LowLFHE.decomp_level_count) -> !LowLFHE.decomp_level_count
func @type_decomp_level_count(%arg0: !LowLFHE.decomp_level_count) -> !LowLFHE.decomp_level_count {
  // CHECK-NEXT: return %arg0 : !LowLFHE.decomp_level_count
  return %arg0: !LowLFHE.decomp_level_count
}

// CHECK-LABEL: func @type_decomp_base_log(%arg0: !LowLFHE.decomp_base_log) -> !LowLFHE.decomp_base_log
func @type_decomp_base_log(%arg0: !LowLFHE.decomp_base_log) -> !LowLFHE.decomp_base_log {
  // CHECK-NEXT: return %arg0 : !LowLFHE.decomp_base_log
  return %arg0: !LowLFHE.decomp_base_log
}

// CHECK-LABEL: func @type_variance(%arg0: !LowLFHE.variance) -> !LowLFHE.variance
func @type_variance(%arg0: !LowLFHE.variance) -> !LowLFHE.variance {
  // CHECK-NEXT: return %arg0 : !LowLFHE.variance
  return %arg0: !LowLFHE.variance
}

// CHECK-LABEL: func @type_cleartext(%arg0: !LowLFHE.cleartext) -> !LowLFHE.cleartext
func @type_cleartext(%arg0: !LowLFHE.cleartext) -> !LowLFHE.cleartext {
  // CHECK-NEXT: return %arg0 : !LowLFHE.cleartext
  return %arg0: !LowLFHE.cleartext
}

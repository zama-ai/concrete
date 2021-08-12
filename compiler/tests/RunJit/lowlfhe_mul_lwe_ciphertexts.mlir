// RUN: zamacompiler %s --run-jit --jit-args 7 --jit-args 3 2>&1| FileCheck %s

// CHECK-LABEL: 21
func @main(%arg0: !LowLFHE.lwe_ciphertext, %arg1: i8) -> !LowLFHE.lwe_ciphertext {
  %cleartext = "LowLFHE.int_to_cleartext"(%arg1) : (i8) -> !LowLFHE.cleartext<64>
  %0 = "LowLFHE.mul_cleartext_lwe_ciphertext"(%arg0, %cleartext) : (!LowLFHE.lwe_ciphertext, !LowLFHE.cleartext<64>) -> !LowLFHE.lwe_ciphertext
  return %0 : !LowLFHE.lwe_ciphertext
}
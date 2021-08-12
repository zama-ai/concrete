// RUN: zamacompiler %s --run-jit --jit-args -32 2>&1| FileCheck %s

// CHECK-LABEL: 32
func @main(%arg0: !LowLFHE.lwe_ciphertext) -> !LowLFHE.lwe_ciphertext {
  %0 = "LowLFHE.negate_lwe_ciphertext"(%arg0) : (!LowLFHE.lwe_ciphertext) -> !LowLFHE.lwe_ciphertext
  return %0 : !LowLFHE.lwe_ciphertext
}
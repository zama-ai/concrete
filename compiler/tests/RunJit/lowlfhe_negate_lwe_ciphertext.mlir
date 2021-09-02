// RUN: zamacompiler %s --run-jit --jit-args 224 2>&1| FileCheck %s

// CHECK-LABEL: 32
func @main(%arg0: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  %0 = "LowLFHE.negate_lwe_ciphertext"(%arg0) : (!LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  return %0 : !LowLFHE.lwe_ciphertext<2048,7>
}
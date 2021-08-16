// RUN: zamacompiler %s --run-jit --jit-args 11 --jit-args 31 2>&1| FileCheck %s

// CHECK-LABEL: 42
func @main(%arg0: !LowLFHE.lwe_ciphertext<2048,7>, %arg1: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  %0 = "LowLFHE.add_lwe_ciphertexts"(%arg0, %arg1) : (!LowLFHE.lwe_ciphertext<2048,7>, !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7>
  return %0 : !LowLFHE.lwe_ciphertext<2048,7>
}
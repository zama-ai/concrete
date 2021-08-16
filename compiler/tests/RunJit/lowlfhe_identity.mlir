// RUN: zamacompiler %s --run-jit --jit-args 42 2>&1| FileCheck %s

// CHECK-LABEL: 42
func @main(%arg0: !LowLFHE.lwe_ciphertext<2048,7>) -> !LowLFHE.lwe_ciphertext<2048,7> {
  return %arg0 : !LowLFHE.lwe_ciphertext<2048,7>
}
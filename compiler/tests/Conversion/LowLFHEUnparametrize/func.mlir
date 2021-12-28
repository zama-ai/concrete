// RUN: concretecompiler --passes lowlfhe-unparametrize --action=dump-llvm-dialect %s  2>&1| FileCheck %s

// CHECK-LABEL: func @main(%arg0: !LowLFHE.lwe_ciphertext<_,_>) -> !LowLFHE.lwe_ciphertext<_,_>
func @main(%arg0: !LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<1024,4> {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_ciphertext<_,_>
  return %arg0: !LowLFHE.lwe_ciphertext<1024,4>
}
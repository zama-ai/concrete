// RUN: concretecompiler --passes lowlfhe-unparametrize --action=dump-llvm-dialect %s  2>&1| FileCheck %s

// CHECK-LABEL: func @main(%arg0: !LowLFHE.lwe_ciphertext<_,_>) -> !LowLFHE.lwe_ciphertext<_,_>
func @main(%arg0: !LowLFHE.lwe_ciphertext<1024,4>) -> !LowLFHE.lwe_ciphertext<_,_> {
  // CHECK-NEXT: return %arg0 : !LowLFHE.lwe_ciphertext<_,_>
  %0 = builtin.unrealized_conversion_cast %arg0 : !LowLFHE.lwe_ciphertext<1024,4> to !LowLFHE.lwe_ciphertext<_,_>
  return %0: !LowLFHE.lwe_ciphertext<_,_>
}
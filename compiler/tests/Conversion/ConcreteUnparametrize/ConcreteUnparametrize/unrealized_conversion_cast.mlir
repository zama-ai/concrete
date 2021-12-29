// RUN: concretecompiler --passes concrete-unparametrize --action=dump-llvm-dialect %s  2>&1| FileCheck %s

// CHECK-LABEL: func @main(%arg0: !Concrete.lwe_ciphertext<_,_>) -> !Concrete.lwe_ciphertext<_,_>
func @main(%arg0: !Concrete.lwe_ciphertext<1024,4>) -> !Concrete.lwe_ciphertext<_,_> {
  // CHECK-NEXT: return %arg0 : !Concrete.lwe_ciphertext<_,_>
  %0 = builtin.unrealized_conversion_cast %arg0 : !Concrete.lwe_ciphertext<1024,4> to !Concrete.lwe_ciphertext<_,_>
  return %0: !Concrete.lwe_ciphertext<_,_>
}
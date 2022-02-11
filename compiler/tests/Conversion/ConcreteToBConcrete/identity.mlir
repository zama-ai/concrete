// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

// CHECK: func @identity(%arg0: tensor<1025xi64>) -> tensor<1025xi64> {
// CHECK-NEXT:   return %arg0 : tensor<1025xi64>
// CHECK-NEXT: }
func @identity(%arg0: !Concrete.lwe_ciphertext<1024,7>) -> !Concrete.lwe_ciphertext<1024,7> {
    return %arg0 : !Concrete.lwe_ciphertext<1024,7>
}
// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s
// CHECK: func @tensor_identity(%arg0: tensor<2x3x4x1025xi64>) -> tensor<2x3x4x1025xi64> {
// CHECK-NEXT:   return %arg0 : tensor<2x3x4x1025xi64>
// CHECK-NEXT: }
func @tensor_identity(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<1024,7>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<1024,7>> {
    return %arg0 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,7>>
}

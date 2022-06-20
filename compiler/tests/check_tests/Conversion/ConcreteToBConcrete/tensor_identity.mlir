// RUN: concretecompiler --passes concrete-to-bconcrete --action=dump-bconcrete %s 2>&1| FileCheck %s

// CHECK: func.func @tensor_identity(%arg0: tensor<2x3x4x1025xi64>) -> tensor<2x3x4x1025xi64> {
// CHECK-NEXT:   return %arg0 : tensor<2x3x4x1025xi64>
// CHECK-NEXT: }
func.func @tensor_identity(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<1024,7>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<1024,7>> {
    return %arg0 : tensor<2x3x4x!Concrete.lwe_ciphertext<1024,7>>
}

// CHECK: func.func @tensor_identity_crt(%arg0: tensor<2x3x4x5x1025xi64>) -> tensor<2x3x4x5x1025xi64> {
// CHECK-NEXT:   return %arg0 : tensor<2x3x4x5x1025xi64>
// CHECK-NEXT: }
func.func @tensor_identity_crt(%arg0: tensor<2x3x4x!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>>) -> tensor<2x3x4x!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>> {
    return %arg0 : tensor<2x3x4x!Concrete.lwe_ciphertext<crt=[2,3,5,7,11],1024,7>>
}

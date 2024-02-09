// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete --skip-program-info %s 2>&1| FileCheck %s

// CHECK: func.func @tensor_identity(%arg0: tensor<2x3x4x1025xi64>) -> tensor<2x3x4x1025xi64> {
// CHECK-NEXT:   return %arg0 : tensor<2x3x4x1025xi64>
// CHECK-NEXT: }
func.func @tensor_identity(%arg0: tensor<2x3x4x!TFHE.glwe<sk[1]<1,1024>>>) -> tensor<2x3x4x!TFHE.glwe<sk[1]<1,1024>>> {
    return %arg0 : tensor<2x3x4x!TFHE.glwe<sk[1]<1,1024>>>
}

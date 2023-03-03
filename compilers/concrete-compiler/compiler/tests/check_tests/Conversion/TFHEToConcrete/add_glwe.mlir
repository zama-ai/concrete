// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_glwe(%arg0: tensor<2049xi64>, %arg1: tensor<2049xi64>) -> tensor<2049xi64>
func.func @add_glwe(%arg0: !TFHE.glwe<{2048,1,64}{7}>, %arg1: !TFHE.glwe<{2048,1,64}{7}>) -> !TFHE.glwe<{2048,1,64}{7}> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.add_lwe_tensor"(%arg0, %arg1) : (tensor<2049xi64>, tensor<2049xi64>) -> tensor<2049xi64>
  // CHECK-NEXT: return %[[V1]] : tensor<2049xi64>

  %0 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<{2048,1,64}{7}>, !TFHE.glwe<{2048,1,64}{7}>) -> (!TFHE.glwe<{2048,1,64}{7}>)
  return %0: !TFHE.glwe<{2048,1,64}{7}>
}

// RUN: concretecompiler --passes tfhe-to-concrete --action=dump-concrete --skip-program-info %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @neg_glwe(%arg0: tensor<1025xi64>) -> tensor<1025xi64>
func.func @neg_glwe(%arg0: !TFHE.glwe<sk[1]<1,1024>>) -> !TFHE.glwe<sk[1]<1,1024>> {
  // CHECK-NEXT: %[[V1:.*]] = "Concrete.negate_lwe_tensor"(%arg0) : (tensor<1025xi64>) -> tensor<1025xi64>
  // CHECK-NEXT: return %[[V1]] : tensor<1025xi64>
  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<sk[1]<1,1024>>) -> (!TFHE.glwe<sk[1]<1,1024>>)
  return %1: !TFHE.glwe<sk[1]<1,1024>>
}

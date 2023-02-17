// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
func.func @add_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>, %arg1: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.add_glwe"(%arg0, %arg1) : (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<sk[1]<12,1024>>

  %0 = "TFHE.add_glwe"(%arg0, %arg1): (!TFHE.glwe<sk[1]<12,1024>>, !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<12,1024>>)
  return %0: !TFHE.glwe<sk[1]<12,1024>>
}

// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @neg_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
func.func @neg_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.neg_glwe"(%arg0) : (!TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<sk[1]<12,1024>>

  %1 = "TFHE.neg_glwe"(%arg0): (!TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<12,1024>>)
  return %1: !TFHE.glwe<sk[1]<12,1024>>
}

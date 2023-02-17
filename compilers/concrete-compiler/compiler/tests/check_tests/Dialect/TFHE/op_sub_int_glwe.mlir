// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @sub_int_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
func.func @sub_int_glwe(%arg0: !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[V2:.*]] = "TFHE.sub_int_glwe"(%[[V1]], %arg0) : (i64, !TFHE.glwe<sk[1]<12,1024>>) -> !TFHE.glwe<sk[1]<12,1024>>
  // CHECK-NEXT: return %[[V2]] : !TFHE.glwe<sk[1]<12,1024>>

  %0 = arith.constant 1 : i64
  %1 = "TFHE.sub_int_glwe"(%0, %arg0): (i64, !TFHE.glwe<sk[1]<12,1024>>) -> (!TFHE.glwe<sk[1]<12,1024>>)
  return %1: !TFHE.glwe<sk[1]<12,1024>>
}

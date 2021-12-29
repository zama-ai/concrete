// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table(%arg0: !TFHE.glwe<{1024,12,64}{7}>, %arg1: tensor<128xi64>) -> !TFHE.glwe<{512,10,64}{2}>
func @apply_lookup_table(%arg0: !TFHE.glwe<{1024,12,64}{7}>, %arg1: tensor<128xi64>) -> !TFHE.glwe<{512,10,64}{2}> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.apply_lookup_table"(%arg0, %arg1) {baseLogBS = -83 : i32, baseLogKS = -82 : i32, glweDimension = 1 : i32, levelBS = 3 : i32, levelKS = 2 : i32, outputSizeKS = 600 : i32, polynomialSize = 1024 : i32} : (!TFHE.glwe<{1024,12,64}{7}>, tensor<128xi64>) -> !TFHE.glwe<{512,10,64}{2}>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<{512,10,64}{2}>

  %1 = "TFHE.apply_lookup_table"(%arg0, %arg1) {glweDimension = 1 : i32, polynomialSize = 1024 : i32, levelKS = 2 : i32, baseLogKS = -82 : i32, levelBS = 3 : i32, baseLogBS = -83 : i32, outputSizeKS = 600 : i32} : (!TFHE.glwe<{1024,12,64}{7}>, tensor<128xi64>) -> (!TFHE.glwe<{512,10,64}{2}>)
  return %1: !TFHE.glwe<{512,10,64}{2}>
}

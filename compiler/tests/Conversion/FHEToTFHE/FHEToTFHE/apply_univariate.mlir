// RUN: concretecompiler %s --passes fhe-to-tfhe --action=dump-tfhe 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table(%arg0: !TFHE.glwe<{_,_,_}{2}>, %arg1: tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{2}>
func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "TFHE.apply_lookup_table"(%arg0, %arg1) {baseLogBS = -1 : i32, baseLogKS = -1 : i32, glweDimension = -1 : i32, levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS = -1 : i32, polynomialSize = -1 : i32} : (!TFHE.glwe<{_,_,_}{2}>, tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{2}>
  // CHECK-NEXT: return %[[V1]] : !TFHE.glwe<{_,_,_}{2}>

  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

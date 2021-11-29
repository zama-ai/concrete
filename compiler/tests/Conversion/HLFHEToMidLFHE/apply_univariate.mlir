// RUN: zamacompiler %s --passes hlfhe-to-midlfhe --action=dump-midlfhe 2>&1| FileCheck %s

// CHECK-LABEL: func @apply_lookup_table(%arg0: !MidLFHE.glwe<{_,_,_}{2}>, %arg1: tensor<4xi64>) -> !MidLFHE.glwe<{_,_,_}{2}>
func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<4xi64>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "MidLFHE.apply_lookup_table"(%arg0, %arg1) {baseLogBS = -1 : i32, baseLogKS = -1 : i32, glweDimension = -1 : i32, levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS = -1 : i32, polynomialSize = -1 : i32} : (!MidLFHE.glwe<{_,_,_}{2}>, tensor<4xi64>) -> !MidLFHE.glwe<{_,_,_}{2}>
  // CHECK-NEXT: return %[[V1]] : !MidLFHE.glwe<{_,_,_}{2}>

  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, tensor<4xi64>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// RUN: concretecompiler %s --passes fhe-to-tfhe --action=dump-tfhe 2>&1| FileCheck %s

// CHECK: func @apply_lookup_table(%[[A0:.*]]: !TFHE.glwe<{_,_,_}{2}>, %[[LUT:.*]]: tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{3}> {
// CHECK-NEXT: %[[V0:.*]] = "TFHE.glwe_from_table"(%[[LUT]]) : (tensor<4xi64>) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT: %[[V1:.*]] = "TFHE.keyswitch_glwe"(%[[A0]]) {baseLog = -1 : i32, level = -1 : i32} : (!TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT: %[[V2:.*]] = "TFHE.bootstrap_glwe"(%[[V1]], %[[V0]]) {baseLog = -1 : i32, glweDimension = -1 : i32, level = -1 : i32, polynomialSize = -1 : i32} : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{3}>
// CHECK-NEXT: return %[[V2]] : !TFHE.glwe<{_,_,_}{3}>
func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<3> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<3>)
  return %1: !FHE.eint<3>
}

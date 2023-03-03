// RUN: concretecompiler %s --action=roundtrip 2>&1 | FileCheck %s

//CHECK: func.func @multi_lut(%[[A0:.*]]: tensor<4x4x!FHE.eint<2>>, %[[A1:.*]]: tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>> {
//CHECK-NEXT:   %[[V0:.*]] = "FHELinalg.apply_multi_lookup_table"(%[[A0]], %[[A1]]) : (tensor<4x4x!FHE.eint<2>>, tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>>
//CHECK-NEXT:   return %[[V0]] : tensor<4x4x!FHE.eint<2>>
//CHECK-NEXT: }
func.func @multi_lut(%arg0: tensor<4x4x!FHE.eint<2>>, %arg1: tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>> {
  %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<4x4x!FHE.eint<2>>, tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>>
  return %1: tensor<4x4x!FHE.eint<2>>
}

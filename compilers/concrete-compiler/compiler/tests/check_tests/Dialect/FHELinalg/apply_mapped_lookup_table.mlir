// RUN: concretecompiler %s --action=roundtrip 2>&1 | FileCheck %s


//CHECK: func.func @mapped_lut(%[[A0:.*]]: tensor<2x3x!FHE.eint<2>>, %[[A1:.*]]: tensor<5x4xi64>, %[[A2:.*]]: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<2>> {
//CHECK-NEXT:   %[[V0:.*]] = "FHELinalg.apply_mapped_lookup_table"(%[[A0]], %[[A1]], %[[A2]]) : (tensor<2x3x!FHE.eint<2>>, tensor<5x4xi64>, tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<2>>
//CHECK-NEXT:   return %[[V0]] : tensor<2x3x!FHE.eint<2>>
//CHECK-NEXT: }
func.func @mapped_lut(%t: tensor<2x3x!FHE.eint<2>>, %luts: tensor<5x4xi64>, %map: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<2>> {
  %0 = "FHELinalg.apply_mapped_lookup_table"(%t, %luts, %map): (tensor<2x3x!FHE.eint<2>>, tensor<5x4xi64>, tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<2>>
  return %0: tensor<2x3x!FHE.eint<2>>
}

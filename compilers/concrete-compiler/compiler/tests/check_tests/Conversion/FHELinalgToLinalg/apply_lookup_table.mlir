// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func.func @apply_lookup_table(%[[Varg0:.*]]: tensor<2x3x4x!FHE.eint<2>>, %[[Varg1:.*]]: tensor<4xi64>) -> tensor<2x3x4x!FHE.eint<2>> {
// CHECK-NEXT:     %[[V0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:     %[[V1:.*]] = linalg.generic {indexing_maps = {{\[}}#map, #map{{\], iterator}}_types = {{\[}}"parallel", "parallel", "parallel"{{\]}}} ins(%[[Varg0]] : tensor<2x3x4x!FHE.eint<2>>) outs(%[[V0]] : tensor<2x3x4x!FHE.eint<2>>) {
// CHECK-NEXT:     ^bb0(%[[Varg2:.*]]: !FHE.eint<2>, %[[Varg3:.*]]: !FHE.eint<2>):
// CHECK-NEXT:       %[[V2:.*]] = "FHE.apply_lookup_table"(%[[Varg2]], %[[Varg1]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
// CHECK-NEXT:       linalg.yield %[[V2]] : !FHE.eint<2>
// CHECK-NEXT:     } -> tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:     return %[[V1]] : tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @apply_lookup_table(%arg0: tensor<2x3x4x!FHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!FHE.eint<2>> {
  %1 = "FHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!FHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

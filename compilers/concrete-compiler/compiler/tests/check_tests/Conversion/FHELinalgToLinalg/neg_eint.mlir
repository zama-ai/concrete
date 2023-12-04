// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func.func @neg_eint(%[[Varg0:.*]]: tensor<2x3x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>> {
// CHECK-NEXT:     %[[V0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:     %[[V1:.*]] = linalg.generic {indexing_maps = {{\[}}#map, #map{{\], iterator}}_types = {{\[}}"parallel", "parallel", "parallel"{{\]}}} ins(%[[Varg0]] : tensor<2x3x4x!FHE.eint<2>>) outs(%[[V0]] : tensor<2x3x4x!FHE.eint<2>>) {
// CHECK-NEXT:     ^bb0(%[[Varg1:.*]]: !FHE.eint<2>, %[[Varg2:.*]]: !FHE.eint<2>):
// CHECK-NEXT:       %[[V2:.*]] = "FHE.neg_eint"(%[[Varg1]]) : (!FHE.eint<2>) -> !FHE.eint<2>
// CHECK-NEXT:       linalg.yield %[[V2]] : !FHE.eint<2>
// CHECK-NEXT:     } -> tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:     return %[[V1]] : tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @neg_eint(%arg0: tensor<2x3x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>> {
  %1 = "FHELinalg.neg_eint"(%arg0): (tensor<2x3x4x!FHE.eint<2>>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

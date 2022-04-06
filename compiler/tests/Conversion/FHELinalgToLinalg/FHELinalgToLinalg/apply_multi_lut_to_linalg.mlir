// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

//CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
//CHECK-NEXT: #map1 = affine_map<(d0, d1) -> (d0, d1, 0)>
//CHECK-NEXT: #map2 = affine_map<(d0, d1) -> (d0, d1, 1)>
//CHECK-NEXT: #map3 = affine_map<(d0, d1) -> (d0, d1, 2)>
//CHECK-NEXT: #map4 = affine_map<(d0, d1) -> (d0, d1, 3)>
//CHECK-NEXT: module  {
//CHECK-NEXT:   func @multi_lut(%[[A0:.*]]: tensor<4x4x!FHE.eint<2>>, %[[A1:.*]]: tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>> {
//CHECK-NEXT:     %[[V0:.*]] = linalg.init_tensor [4, 4] : tensor<4x4x!FHE.eint<2>>
//CHECK-NEXT:     %[[V1:.*]] = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3, #map4, #map0], iterator_types = ["parallel", "parallel"]} ins(%[[A0]], %[[A1]], %[[A1]], %[[A1]], %[[A1]] : tensor<4x4x!FHE.eint<2>>, tensor<4x4x4xi64>, tensor<4x4x4xi64>, tensor<4x4x4xi64>, tensor<4x4x4xi64>) outs(%[[V0]] : tensor<4x4x!FHE.eint<2>>) {
//CHECK-NEXT:     ^bb0(%[[A2:.*]]: !FHE.eint<2>, %[[A3:.*]]: i64, %[[A4:.*]]: i64, %[[A5:.*]]: i64, %[[A6:.*]]: i64, %[[A7:.*]]: !FHE.eint<2>):  // no predecessors
//CHECK-NEXT:       %[[V2:.*]] = tensor.from_elements %[[A3]], %[[A4]], %[[A5]], %[[A6]] : tensor<4xi64>
//CHECK-NEXT:       %[[V3:.*]] = "FHE.apply_lookup_table"(%[[A2]], %[[V2]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
//CHECK-NEXT:       linalg.yield %[[V3]] : !FHE.eint<2>
//CHECK-NEXT:     } -> tensor<4x4x!FHE.eint<2>>
//CHECK-NEXT:     return %[[V1]] : tensor<4x4x!FHE.eint<2>>
//CHECK-NEXT:   }
//CHECK-NEXT: }

func @multi_lut(%arg0: tensor<4x4x!FHE.eint<2>>, %arg1: tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>> {
  %0 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<4x4x!FHE.eint<2>>, tensor<4x4x4xi64>) -> tensor<4x4x!FHE.eint<2>>
  return %0: tensor<4x4x!FHE.eint<2>>
}
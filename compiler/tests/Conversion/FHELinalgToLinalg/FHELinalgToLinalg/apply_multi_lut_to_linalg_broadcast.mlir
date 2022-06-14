// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

//CHECK: #map0 = affine_map<(d0, d1) -> (d0, d1)>
//CHECK: #map1 = affine_map<(d0, d1) -> (d1, 0)>
//CHECK: #map2 = affine_map<(d0, d1) -> (d1, 1)>
//CHECK: #map3 = affine_map<(d0, d1) -> (d1, 2)>
//CHECK: #map4 = affine_map<(d0, d1) -> (d1, 3)>
//CHECK: func @multi_lut(%[[A0:.*]]: tensor<4x3x!FHE.eint<2>>, %[[A1:.*]]: tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>> {
//CHECK:   %[[V0:.*]] = linalg.init_tensor [4, 3] : tensor<4x3x!FHE.eint<2>>
//CHECK:   %[[V1:.*]] = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3, #map4, #map0], iterator_types = ["parallel", "parallel"]} ins(%[[A0]], %[[A1]], %arg1, %arg1, %arg1 : tensor<4x3x!FHE.eint<2>>, tensor<3x4xi64>, tensor<3x4xi64>, tensor<3x4xi64>, tensor<3x4xi64>) outs(%[[V0]] : tensor<4x3x!FHE.eint<2>>) {
//CHECK:   ^bb0(%[[A2:.*]]: !FHE.eint<2>, %[[A3:.*]]: i64, %[[A4:.*]]: i64, %[[A5:.*]]: i64, %[[A6:.*]]: i64, %[[A7:.*]]: !FHE.eint<2>):
//CHECK:     %[[V2:.*]] = tensor.from_elements %[[A3]], %[[A4]], %[[A5]], %[[A6]] : tensor<4xi64>
//CHECK:     %[[V3:.*]] = "FHE.apply_lookup_table"(%[[A2]], %[[V2]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
//CHECK:     linalg.yield %[[V3]] : !FHE.eint<2>
//CHECK:   } -> tensor<4x3x!FHE.eint<2>>
//CHECK:   return %[[V1]] : tensor<4x3x!FHE.eint<2>>
//CHECK: }
func @multi_lut(%arg0: tensor<4x3x!FHE.eint<2>>, %arg1: tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>> {
  %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<4x3x!FHE.eint<2>>, tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}

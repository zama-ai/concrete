// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

//CHECK: #map = affine_map<(d0, d1) -> (d0, d1)>
//CHECK: func.func @multi_lut(%arg0: tensor<4x3x!FHE.eint<2>>, %[[LUTS:.*]]: tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>> {
//CHECK:   %[[MEM:.*]] = "FHE.zero_tensor"() : () -> tensor<4x3x!FHE.eint<2>>
//CHECK:   %[[R:.*]] = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0 : tensor<4x3x!FHE.eint<2>>) outs(%[[MEM]] : tensor<4x3x!FHE.eint<2>>) {
//CHECK:   ^bb0(%[[IN:.*]]: !FHE.eint<2>, %[[UNUSED:.*]]: !FHE.eint<2>):
//CHECK:     %[[INDEX:.*]] = linalg.index 1 : index
//CHECK:     %[[LUT:.*]] = tensor.extract_slice %arg1[%[[INDEX]], 0] [1, 4] [1, 1] : tensor<3x4xi64> to tensor<4xi64>
//CHECK:     %[[V:.*]] = "FHE.apply_lookup_table"(%[[IN]], %[[LUT]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
//CHECK:     linalg.yield %[[V]] : !FHE.eint<2>
//CHECK:   } -> tensor<4x3x!FHE.eint<2>>
//CHECK:   return %[[R]] : tensor<4x3x!FHE.eint<2>>
//CHECK: }
func.func @multi_lut(%arg0: tensor<4x3x!FHE.eint<2>>, %luts: tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>> {
  %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %luts): (tensor<4x3x!FHE.eint<2>>, tensor<3x4xi64>) -> tensor<4x3x!FHE.eint<2>>
  return %1: tensor<4x3x!FHE.eint<2>>
}

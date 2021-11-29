// RUN: zamacompiler %s --action=dump-midlfhe 2>&1 | FileCheck %s

//CHECK-LABEL: #map0 = affine_map<(d0, d1) -> (d0, d1)>
//CHECK-NEXT: #map1 = affine_map<(d0, d1) -> (d0, d1, 0)>
//CHECK-NEXT: #map2 = affine_map<(d0, d1) -> (d0, d1, 1)>
//CHECK-NEXT: #map3 = affine_map<(d0, d1) -> (d0, d1, 2)>
//CHECK-NEXT: #map4 = affine_map<(d0, d1) -> (d0, d1, 3)>
//CHECK-NEXT: module  {
//CHECK-NEXT:   func @multi_lut(%arg0: tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<4x4x4xi64>) -> tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>> {
//CHECK-NEXT:     %[[V0:.*]] = linalg.init_tensor [4, 4] : tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %[[V1:.*]] = linalg.generic {indexing_maps = [#map0, #map1, #map2, #map3, #map4, #map0], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1, %arg1, %arg1, %arg1 : tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>>, tensor<4x4x4xi64>, tensor<4x4x4xi64>, tensor<4x4x4xi64>, tensor<4x4x4xi64>) outs(%[[V0]] : tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:     ^bb0(%arg2: !MidLFHE.glwe<{_,_,_}{2}>, %arg3: i64, %arg4: i64, %arg5: i64, %arg6: i64, %arg7: !MidLFHE.glwe<{_,_,_}{2}>):  // no predecessors
//CHECK-NEXT:       %[[V2:.*]] = tensor.from_elements %arg3, %arg4, %arg5, %arg6 : tensor<4xi64>
//CHECK-NEXT:       %[[V3:.*]] = "MidLFHE.apply_lookup_table"(%arg2, %[[V2]]) {baseLogBS = -1 : i32, baseLogKS = -1 : i32, glweDimension = -1 : i32, levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS = -1 : i32, polynomialSize = -1 : i32} : (!MidLFHE.glwe<{_,_,_}{2}>, tensor<4xi64>) -> !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       linalg.yield %[[V3]] : !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     } -> tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     return %[[V1]] : tensor<4x4x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:   }
//CHECK-NEXT: }
func @multi_lut(%arg0: tensor<4x4x!HLFHE.eint<2>>, %arg1: tensor<4x4x4xi64>) -> tensor<4x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.apply_multi_lookup_table"(%arg0, %arg1): (tensor<4x4x!HLFHE.eint<2>>, tensor<4x4x4xi64>) -> tensor<4x4x!HLFHE.eint<2>>
  return %1: tensor<4x4x!HLFHE.eint<2>>
}
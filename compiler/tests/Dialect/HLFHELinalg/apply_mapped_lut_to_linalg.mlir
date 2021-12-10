// RUN: zamacompiler %s --action=dump-midlfhe 2>&1 | FileCheck %s


//CHECK-LABEL: #map = affine_map<(d0, d1) -> (d0, d1)>
//CHECK-NEXT:module  {
//CHECK-NEXT:  func @mapped_lut(%arg0: tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>>, %[[LUTS:.*]]: tensor<5x4xi64>, %arg2: tensor<2x3xindex>) -> tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>> {
//CHECK-NEXT:     %[[V0:.*]] = linalg.init_tensor [2, 3] : tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %[[V1:.*]] = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg2 : tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>>, tensor<2x3xindex>) outs(%0 : tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:     ^bb0(%arg3: !MidLFHE.glwe<{_,_,_}{2}>, %[[LUTIDX:.*]]: index, %arg5: !MidLFHE.glwe<{_,_,_}{2}>):  // no predecessors
//DISABLED-CHECK-NEXT:       %[[V3:.*]] = tensor.extract_slice %arg1[%[[LUTIDX]], 0] [1, 4] [1, 1] : tensor<5x4xi64> to tensor<4xi64>
//WORKAROUND BEGIN
//CHECK-NEXT:       %[[C0:.*]] = arith.constant 0 : index
//CHECK-NEXT:       %[[C1:.*]] = arith.constant 1 : index
//CHECK-NEXT:       %[[C2:.*]] = arith.constant 2 : index
//CHECK-NEXT:       %[[C3:.*]] = arith.constant 3 : index
//CHECK-NEXT:       %[[E0:.*]] = tensor.extract %[[LUTS]][%[[LUTIDX]], %[[C0]]] : tensor<5x4xi64>
//CHECK-NEXT:       %[[E1:.*]] = tensor.extract %[[LUTS]][%[[LUTIDX]], %[[C1]]] : tensor<5x4xi64>
//CHECK-NEXT:       %[[E2:.*]] = tensor.extract %[[LUTS]][%[[LUTIDX]], %[[C2]]] : tensor<5x4xi64>
//CHECK-NEXT:       %[[E3:.*]] = tensor.extract %[[LUTS]][%[[LUTIDX]], %[[C3]]] : tensor<5x4xi64>
//CHECK-NEXT:       %[[LUT:.*]] = tensor.from_elements %[[E0]], %[[E1]], %[[E2]], %[[E3]] : tensor<4xi64>
//WORKAROUND END
//CHECK-NEXT:       %[[V4:.*]] = "MidLFHE.apply_lookup_table"(%arg3, %[[LUT]]) {baseLogBS = -1 : i32, baseLogKS = -1 : i32, glweDimension = -1 : i32, levelBS = -1 : i32, levelKS = -1 : i32, outputSizeKS = -1 : i32, polynomialSize = -1 : i32} : (!MidLFHE.glwe<{_,_,_}{2}>, tensor<4xi64>) -> !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       linalg.yield %[[V4]] : !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     } -> tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     return %[[V1]] : tensor<2x3x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:   }
//CHECK-NEXT: }
func @mapped_lut(%t: tensor<2x3x!HLFHE.eint<2>>, %luts: tensor<5x4xi64>, %map: tensor<2x3xindex>) -> tensor<2x3x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.apply_mapped_lookup_table"(%t, %luts, %map): (tensor<2x3x!HLFHE.eint<2>>, tensor<5x4xi64>, tensor<2x3xindex>) -> tensor<2x3x!HLFHE.eint<2>>
  return %1: tensor<2x3x!HLFHE.eint<2>>
}

// RUN: concretecompiler %s --action=dump-midlfhe --passes hlfhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @apply_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
// CHECK-NEXT:     %0 = linalg.init_tensor [2, 3, 4] : tensor<2x3x4x!HLFHE.eint<2>>
// CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x3x4x!HLFHE.eint<2>>) outs(%0 : tensor<2x3x4x!HLFHE.eint<2>>) {
// CHECK-NEXT:     ^bb0(%arg2: !HLFHE.eint<2>, %arg3: !HLFHE.eint<2>):  // no predecessors
// CHECK-NEXT:       %2 = "HLFHE.apply_lookup_table"(%arg2, %arg1) : (!HLFHE.eint<2>, tensor<4xi64>) -> !HLFHE.eint<2>
// CHECK-NEXT:       linalg.yield %2 : !HLFHE.eint<2>
// CHECK-NEXT:     } -> tensor<2x3x4x!HLFHE.eint<2>>
// CHECK-NEXT:     return %1 : tensor<2x3x4x!HLFHE.eint<2>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func @apply_lookup_table(%arg0: tensor<2x3x4x!HLFHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!HLFHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!HLFHE.eint<2>>)
  return %1: tensor<2x3x4x!HLFHE.eint<2>>
}
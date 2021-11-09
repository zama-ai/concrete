// RUN: zamacompiler %s --action=dump-midlfhe --passes hlfhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
// CHECK-NEXT:     %0 = tensor.generate   {
// CHECK-NEXT:     ^bb0(%arg2: index, %arg3: index):  // no predecessors
// CHECK-NEXT:       %2 = "HLFHE.zero"() : () -> !HLFHE.eint<2>
// CHECK-NEXT:       tensor.yield %2 : !HLFHE.eint<2>
// CHECK-NEXT:     } : tensor<3x2x!HLFHE.eint<2>>
// CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) outs(%0 : tensor<3x2x!HLFHE.eint<2>>) {
// CHECK-NEXT:     ^bb0(%arg2: !HLFHE.eint<2>, %arg3: i3, %arg4: !HLFHE.eint<2>):  // no predecessors
// CHECK-NEXT:       %2 = "HLFHE.mul_eint_int"(%arg2, %arg3) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
// CHECK-NEXT:       %3 = "HLFHE.add_eint"(%arg4, %2) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
// CHECK-NEXT:       linalg.yield %3 : !HLFHE.eint<2>
// CHECK-NEXT:     } -> tensor<3x2x!HLFHE.eint<2>>
// CHECK-NEXT:     return %1 : tensor<3x2x!HLFHE.eint<2>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func @matmul_eint_int(%arg0: tensor<3x4x!HLFHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>> {
  %1 = "HLFHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x4x!HLFHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!HLFHE.eint<2>>
  return %1 : tensor<3x2x!HLFHE.eint<2>>
}
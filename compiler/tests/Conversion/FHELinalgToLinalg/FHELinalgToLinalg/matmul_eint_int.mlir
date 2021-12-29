// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map0 = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @matmul_eint_int(%arg0: tensor<3x4x!FHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
// CHECK-NEXT:     %0 = tensor.generate   {
// CHECK-NEXT:     ^bb0(%arg2: index, %arg3: index):  // no predecessors
// CHECK-NEXT:       %2 = "FHE.zero"() : () -> !FHE.eint<2>
// CHECK-NEXT:       tensor.yield %2 : !FHE.eint<2>
// CHECK-NEXT:     } : tensor<3x2x!FHE.eint<2>>
// CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map0, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<3x4x!FHE.eint<2>>, tensor<4x2xi3>) outs(%0 : tensor<3x2x!FHE.eint<2>>) {
// CHECK-NEXT:     ^bb0(%arg2: !FHE.eint<2>, %arg3: i3, %arg4: !FHE.eint<2>):  // no predecessors
// CHECK-NEXT:       %2 = "FHE.mul_eint_int"(%arg2, %arg3) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
// CHECK-NEXT:       %3 = "FHE.add_eint"(%arg4, %2) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
// CHECK-NEXT:       linalg.yield %3 : !FHE.eint<2>
// CHECK-NEXT:     } -> tensor<3x2x!FHE.eint<2>>
// CHECK-NEXT:     return %1 : tensor<3x2x!FHE.eint<2>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func @matmul_eint_int(%arg0: tensor<3x4x!FHE.eint<2>>, %arg1: tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>> {
  %1 = "FHELinalg.matmul_eint_int"(%arg0, %arg1): (tensor<3x4x!FHE.eint<2>>, tensor<4x2xi3>) -> tensor<3x2x!FHE.eint<2>>
  return %1 : tensor<3x2x!FHE.eint<2>>
}
// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK:      #map = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func.func @main(%arg0: tensor<2x3x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.esint<2>> {
// CHECK-NEXT:     %0 = bufferization.alloc_tensor() : tensor<2x3x4x!FHE.esint<2>>
// CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<2x3x4x!FHE.eint<2>>) outs(%0 : tensor<2x3x4x!FHE.esint<2>>) {
// CHECK-NEXT:     ^bb0(%arg1: !FHE.eint<2>, %arg2: !FHE.esint<2>):
// CHECK-NEXT:       %2 = "FHE.to_signed"(%arg1) : (!FHE.eint<2>) -> !FHE.esint<2>
// CHECK-NEXT:       linalg.yield %2 : !FHE.esint<2>
// CHECK-NEXT:     } -> tensor<2x3x4x!FHE.esint<2>>
// CHECK-NEXT:     return %1 : tensor<2x3x4x!FHE.esint<2>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x3x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.esint<2>> {
  %1 = "FHELinalg.to_signed"(%arg0): (tensor<2x3x4x!FHE.eint<2>>) -> (tensor<2x3x4x!FHE.esint<2>>)
  return %1: tensor<2x3x4x!FHE.esint<2>>
}

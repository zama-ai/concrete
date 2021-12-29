// RUN: concretecompiler %s --passes fhe-to-tfhe --action=dump-tfhe 2>&1| FileCheck %s

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> (0)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @linalg_generic(%arg0: tensor<2x!TFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<2xi3>, %arg2: tensor<1x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:     %0 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!TFHE.glwe<{_,_,_}{2}>>, tensor<2xi3>) outs(%arg2 : tensor<1x!TFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:     ^bb0(%arg3: !TFHE.glwe<{_,_,_}{2}>, %arg4: i3, %arg5: !TFHE.glwe<{_,_,_}{2}>):  // no predecessors
// CHECK-NEXT:       %1 = "TFHE.mul_glwe_int"(%arg3, %arg4) : (!TFHE.glwe<{_,_,_}{2}>, i3) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:       %2 = "TFHE.add_glwe"(%1, %arg5) : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:       linalg.yield %2 : !TFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:     } -> tensor<1x!TFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
module  {
  func @linalg_generic(%arg0: tensor<2x!FHE.eint<2>>, %arg1: tensor<2xi3>, %acc: tensor<1x!FHE.eint<2>>) {
    %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!FHE.eint<2>>, tensor<2xi3>) outs(%acc : tensor<1x!FHE.eint<2>>) {
    ^bb0(%arg2: !FHE.eint<2>, %arg3: i3, %arg4: !FHE.eint<2>):  // no predecessors
      %4 = "FHE.mul_eint_int"(%arg2, %arg3) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
      %5 = "FHE.add_eint"(%4, %arg4) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
      linalg.yield %5 : !FHE.eint<2>
    } -> tensor<1x!FHE.eint<2>>
    return
  }
}

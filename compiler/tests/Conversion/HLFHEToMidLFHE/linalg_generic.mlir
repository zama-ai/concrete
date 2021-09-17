// RUN: zamacompiler %s --entry-dialect=hlfhe --action=dump-midlfhe 2>&1| FileCheck %s

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> (0)>
// CHECK-NEXT: module  {
// CHECK-NEXT:   func @linalg_generic(%arg0: tensor<2x!MidLFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<2xi3>, %arg2: tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:     %0 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!MidLFHE.glwe<{_,_,_}{2}>>, tensor<2xi3>) outs(%arg2 : tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT:     ^bb0(%arg3: !MidLFHE.glwe<{_,_,_}{2}>, %arg4: i3, %arg5: !MidLFHE.glwe<{_,_,_}{2}>):  // no predecessors
// CHECK-NEXT:       %1 = "MidLFHE.mul_glwe_int"(%arg3, %arg4) : (!MidLFHE.glwe<{_,_,_}{2}>, i3) -> !MidLFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:       %2 = "MidLFHE.add_glwe"(%1, %arg5) : (!MidLFHE.glwe<{_,_,_}{2}>, !MidLFHE.glwe<{_,_,_}{2}>) -> !MidLFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:       linalg.yield %2 : !MidLFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT:     } -> tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> (0)>
module  {
  func @linalg_generic(%arg0: tensor<2x!HLFHE.eint<2>>, %arg1: tensor<2xi3>, %acc: tensor<1x!HLFHE.eint<2>>) {
    %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) outs(%acc : tensor<1x!HLFHE.eint<2>>) {
    ^bb0(%arg2: !HLFHE.eint<2>, %arg3: i3, %arg4: !HLFHE.eint<2>):  // no predecessors
      %4 = "HLFHE.mul_eint_int"(%arg2, %arg3) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
      %5 = "HLFHE.add_eint"(%4, %arg4) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
      linalg.yield %5 : !HLFHE.eint<2>
    } -> tensor<1x!HLFHE.eint<2>>
    return
  }
}

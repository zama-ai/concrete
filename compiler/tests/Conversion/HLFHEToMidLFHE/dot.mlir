// RUN: zamacompiler %s --passes hlfhe-to-midlfhe 2>&1| FileCheck %s

//CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT #map1 = affine_map<(d0) -> ()>
// CHECK-NEXT module  {
// CHECK-NEXT   func @dot_eint_int(%arg0: memref<2x!MidLFHE.glwe<{_,_,_}{2}>>, %arg1: memref<2xi3>, %arg2: memref<!MidLFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT     linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : memref<2x!MidLFHE.glwe<{_,_,_}{2}>>, memref<2xi3>) outs(%arg2 : memref<!MidLFHE.glwe<{_,_,_}{2}>>) {
// CHECK-NEXT     ^bb0(%arg3: !MidLFHE.glwe<{_,_,_}{2}>, %arg4: i3, %arg5: !MidLFHE.glwe<{_,_,_}{2}>):  // no predecessors
// CHECK-NEXT       %0 = "MidLFHE.mul_glwe_int"(%arg3, %arg4) : (!MidLFHE.glwe<{_,_,_}{2}>, i3) -> !MidLFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT       %1 = "MidLFHE.add_glwe"(%0, %arg5) : (!MidLFHE.glwe<{_,_,_}{2}>, !MidLFHE.glwe<{_,_,_}{2}>) -> !MidLFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT       linalg.yield %1 : !MidLFHE.glwe<{_,_,_}{2}>
// CHECK-NEXT     }
// CHECK-NEXT     return
// CHECK-NEXT   }
// CHECK-NEXT }

#map0 = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0) -> ()>
module  {
  func @dot_eint_int(%arg0: memref<2x!HLFHE.eint<2>>, %arg1: memref<2xi3>, %arg2: memref<!HLFHE.eint<2>>) {
    linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : memref<2x!HLFHE.eint<2>>, memref<2xi3>) outs(%arg2 : memref<!HLFHE.eint<2>>) {
    ^bb0(%arg3: !HLFHE.eint<2>, %arg4: i3, %arg5: !HLFHE.eint<2>):  // no predecessors
      %0 = "HLFHE.mul_eint_int"(%arg3, %arg4) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
      %1 = "HLFHE.add_eint"(%0, %arg5) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
      linalg.yield %1 : !HLFHE.eint<2>
    }
    return
  }
}

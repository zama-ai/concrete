// RUN: zamacompiler %s --passes hlfhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
// CHECK-NEXT: module {
// CHECK-NEXT: func @dot_eint_int(%[[A0:.*]]: memref<2x!HLFHE.eint<2>>, %[[A1:.*]]: memref<2xi3>, %[[A2:.*]]: memref<!HLFHE.eint<2>>)
// CHECK-NEXT: linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%[[A0]], %[[A1]] : memref<2x!HLFHE.eint<2>>, memref<2xi3>) outs(%arg2 : memref<!HLFHE.eint<2>>) {
// CHECK-NEXT:   ^bb0(%[[A3:.*]]: !HLFHE.eint<2>, %[[A4:.*]]: i3, %[[A5:.*]]: !HLFHE.eint<2>):  // no predecessors
// CHECK-NEXT:     %[[T0:.*]] = "HLFHE.mul_eint_int"(%[[A3]], %[[A4]]) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
// CHECK-NEXT:     %[[T1:.*]] = "HLFHE.add_eint"(%[[T0]], %[[A5]]) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
// CHECK-NEXT:     linalg.yield %[[T1]] : !HLFHE.eint<2>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT: }
func @dot_eint_int(%arg0: memref<2x!HLFHE.eint<2>>,
          %arg1: memref<2xi3>,
          %arg2: memref<!HLFHE.eint<2>>)
{
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<2x!HLFHE.eint<2>>, memref<2xi3>, memref<!HLFHE.eint<2>>) -> ()
  return
}

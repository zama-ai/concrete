// RUN: zamacompiler %s --convert-hlfhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #map0 = affine_map<(d0) -> (d0)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> ()>
// CHECK-NEXT: module {
// CHECK-NEXT: func @dot_eint_int(%[[A0:.*]]: memref<2x!HLFHE.eint<0>>, %[[A1:.*]]: memref<2xi32>, %[[A2:.*]]: memref<!HLFHE.eint<0>>)
// CHECK-NEXT: linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%[[A0]], %[[A1]] : memref<2x!HLFHE.eint<0>>, memref<2xi32>) outs(%arg2 : memref<!HLFHE.eint<0>>) {
// CHECK-NEXT:   ^bb0(%[[A3:.*]]: !HLFHE.eint<0>, %[[A4:.*]]: i32, %[[A5:.*]]: !HLFHE.eint<0>):  // no predecessors
// CHECK-NEXT:     %[[T0:.*]] = "HLFHE.mul_eint_int"(%[[A3]], %[[A4]]) : (!HLFHE.eint<0>, i32) -> !HLFHE.eint<0>
// CHECK-NEXT:     %[[T1:.*]] = "HLFHE.add_eint"(%[[T0]], %[[A5]]) : (!HLFHE.eint<0>, !HLFHE.eint<0>) -> !HLFHE.eint<0>
// CHECK-NEXT:     linalg.yield %[[T1]] : !HLFHE.eint<0>
// CHECK-NEXT:  }
// CHECK-NEXT:  return
// CHECK-NEXT: }
func @dot_eint_int(%arg0: memref<2x!HLFHE.eint<0>>,
          %arg1: memref<2xi32>,
          %arg2: memref<!HLFHE.eint<0>>)
{
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<2x!HLFHE.eint<0>>, memref<2xi32>, memref<!HLFHE.eint<0>>) -> ()
  return
}

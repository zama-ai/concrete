// RUN: zamacompiler %s --passes hlfhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

//CHECK: #map0 = affine_map<(d0) -> (d0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (0)>
//CHECK-NEXT: module  {
//CHECK-NEXT:   func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>, %arg1: tensor<2xi3>) -> !HLFHE.eint<2> {
//CHECK-NEXT:     %0 = "HLFHE.zero"() : () -> !HLFHE.eint<2>
//CHECK-NEXT:     %1 = tensor.from_elements %0 : tensor<1x!HLFHE.eint<2>>
//CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) outs(%1 : tensor<1x!HLFHE.eint<2>>) {
//CHECK-NEXT:     ^bb0(%arg2: !HLFHE.eint<2>, %arg3: i3, %arg4: !HLFHE.eint<2>):  // no predecessors
//CHECK-NEXT:       %4 = "HLFHE.mul_eint_int"(%arg2, %arg3) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
//CHECK-NEXT:       %5 = "HLFHE.add_eint"(%4, %arg4) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
//CHECK-NEXT:       linalg.yield %5 : !HLFHE.eint<2>
//CHECK-NEXT:     } -> tensor<1x!HLFHE.eint<2>>
//CHECK-NEXT:     %c0 = constant 0 : index
//CHECK-NEXT:     %3 = tensor.extract %2[%c0] : tensor<1x!HLFHE.eint<2>>
//CHECK-NEXT:     return %3 : !HLFHE.eint<2>
//CHECK-NEXT:   }
//CHECK-NEXT:  }
func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
{
  %o = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>
  return %o : !HLFHE.eint<2>
}

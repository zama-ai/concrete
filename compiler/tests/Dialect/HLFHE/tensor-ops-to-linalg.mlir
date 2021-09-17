// RUN: zamacompiler %s --entry-dialect=hlfhe --action=dump-midlfhe 2>&1 | FileCheck %s

//CHECK: #map0 = affine_map<(d0) -> (d0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (0)>
//CHECK-NEXT: module  {
//CHECK-NEXT:   func @dot_eint_int(%arg0: tensor<2x!MidLFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<2xi3>) -> !MidLFHE.glwe<{_,_,_}{2}> {
//CHECK-NEXT:     %0 = "MidLFHE.zero"() : () -> !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     %1 = tensor.from_elements %0 : tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!MidLFHE.glwe<{_,_,_}{2}>>, tensor<2xi3>) outs(%1 : tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:     ^bb0(%arg2: !MidLFHE.glwe<{_,_,_}{2}>, %arg3: i3, %arg4: !MidLFHE.glwe<{_,_,_}{2}>):  // no predecessors
//CHECK-NEXT:       %4 = "MidLFHE.mul_glwe_int"(%arg2, %arg3) : (!MidLFHE.glwe<{_,_,_}{2}>, i3) -> !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       %5 = "MidLFHE.add_glwe"(%4, %arg4) : (!MidLFHE.glwe<{_,_,_}{2}>, !MidLFHE.glwe<{_,_,_}{2}>) -> !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       linalg.yield %5 : !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     } -> tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %c0 = constant 0 : index
//CHECK-NEXT:     %3 = tensor.extract %2[%c0] : tensor<1x!MidLFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     return %3 : !MidLFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:   }
//CHECK-NEXT: }
func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
{
  %o = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>
  return %o : !HLFHE.eint<2>
}

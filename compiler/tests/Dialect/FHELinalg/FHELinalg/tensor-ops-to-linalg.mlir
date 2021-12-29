// RUN: concretecompiler %s --action=dump-tfhe 2>&1 | FileCheck %s

//CHECK: #map0 = affine_map<(d0) -> (d0)>
//CHECK-NEXT: #map1 = affine_map<(d0) -> (0)>
//CHECK-NEXT: module  {
//CHECK-NEXT:   func @dot_eint_int(%arg0: tensor<2x!TFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<2xi3>) -> !TFHE.glwe<{_,_,_}{2}> {
//CHECK-NEXT:     %0 = "TFHE.zero"() : () -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     %1 = tensor.from_elements %0 : tensor<1x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map0, #map0, #map1], iterator_types = ["reduction"]} ins(%arg0, %arg1 : tensor<2x!TFHE.glwe<{_,_,_}{2}>>, tensor<2xi3>) outs(%1 : tensor<1x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:     ^bb0(%arg2: !TFHE.glwe<{_,_,_}{2}>, %arg3: i3, %arg4: !TFHE.glwe<{_,_,_}{2}>):  // no predecessors
//CHECK-NEXT:       %4 = "TFHE.mul_glwe_int"(%arg2, %arg3) : (!TFHE.glwe<{_,_,_}{2}>, i3) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       %5 = "TFHE.add_glwe"(%4, %arg4) : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       linalg.yield %5 : !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     } -> tensor<1x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %c0 = arith.constant 0 : index
//CHECK-NEXT:     %3 = tensor.extract %2[%c0] : tensor<1x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     return %3 : !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:   }
//CHECK-NEXT: }
func @dot_eint_int(%arg0: tensor<2x!FHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !FHE.eint<2>
{
  %o = "FHELinalg.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!FHE.eint<2>>, tensor<2xi3>) -> !FHE.eint<2>
  return %o : !FHE.eint<2>
}

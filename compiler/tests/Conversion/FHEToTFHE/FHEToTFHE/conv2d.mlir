// RUN: concretecompiler %s --action=dump-tfhe 2>&1| FileCheck %s

//CHECK: #map0 = affine_map<(d0, d1, d2, d3) -> (d1)>
//CHECK-NEXT: #map1 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
//CHECK-NEXT: #map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d4, d2 + d5, d3 + d6)>
//CHECK-NEXT: #map3 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d1, d4, d5, d6)>
//CHECK-NEXT: #map4 = affine_map<(d0, d1, d2, d3, d4, d5, d6) -> (d0, d1, d2, d3)>
//CHECK-NEXT: module  {
//CHECK-NEXT:   func @conv2d(%arg0: tensor<100x3x28x28x!TFHE.glwe<{_,_,_}{2}>>, %arg1: tensor<4x3x14x14xi3>, %arg2: tensor<4xi3>) -> tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>> {
//CHECK-NEXT:     %0 = "TFHE.zero_tensor"() : () -> tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %1 = linalg.generic {indexing_maps = [#map0, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg2 : tensor<4xi3>) outs(%0 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:     ^bb0(%arg3: i3, %arg4: !TFHE.glwe<{_,_,_}{2}>):
//CHECK-NEXT:       %3 = "TFHE.add_glwe_int"(%arg4, %arg3) : (!TFHE.glwe<{_,_,_}{2}>, i3) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       linalg.yield %3 : !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     } -> tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     %2 = linalg.generic {indexing_maps = [#map2, #map3, #map4], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%arg0, %arg1 : tensor<100x3x28x28x!TFHE.glwe<{_,_,_}{2}>>, tensor<4x3x14x14xi3>) outs(%1 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>) {
//CHECK-NEXT:     ^bb0(%arg3: !TFHE.glwe<{_,_,_}{2}>, %arg4: i3, %arg5: !TFHE.glwe<{_,_,_}{2}>):
//CHECK-NEXT:       %3 = "TFHE.mul_glwe_int"(%arg3, %arg4) : (!TFHE.glwe<{_,_,_}{2}>, i3) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       %4 = "TFHE.add_glwe"(%arg5, %3) : (!TFHE.glwe<{_,_,_}{2}>, !TFHE.glwe<{_,_,_}{2}>) -> !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:       linalg.yield %4 : !TFHE.glwe<{_,_,_}{2}>
//CHECK-NEXT:     } -> tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:     return %2 : tensor<100x4x15x15x!TFHE.glwe<{_,_,_}{2}>>
//CHECK-NEXT:   }
//CHECK-NEXT: }

func @conv2d(%input: tensor<100x3x28x28x!FHE.eint<2>>, %weight: tensor<4x3x14x14xi3>, %bias: tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>> {
  %1 = "FHELinalg.conv2d"(%input, %weight, %bias){strides = dense<[1,1]> : tensor<2xi64>, dilations = dense<[1,1]> : tensor<2xi64>, padding = dense<[0, 0, 0, 0]> : tensor<4xi64>}: (tensor<100x3x28x28x!FHE.eint<2>>, tensor<4x3x14x14xi3>, tensor<4xi3>) -> tensor<100x4x15x15x!FHE.eint<2>>
  return %1 : tensor<100x4x15x15x!FHE.eint<2>>
}

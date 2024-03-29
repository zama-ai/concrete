// RUN: concretecompiler --action=dump-fhe-no-linalg %s 2>&1 --optimizer-strategy=dag-mono --split-input-file | FileCheck %s

// CHECK:     %[[V4:.*]] = scf.forall (%[[Varg2:.*]]) in (2) shared_outs(%[[Varg3:.*]] = %[[V3:.*]]) -> (tensor<8x2x2x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x2x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       %[[V6:.*]] = affine.apply #map1(%[[Varg2]])
// CHECK-NEXT:       %[[V7:.*]] = affine.apply #map1(%[[Varg2]])
// CHECK-NEXT:       %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg0:.*]]{{\[0,}} %[[V6]]{{\] \[8, 2\] \[1, 1\]}} : tensor<8x4x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       %[[Vextracted_slice_1:.*]] = tensor.extract_slice %[[Varg1:.*]]{{\[}}%[[V7]], 0{{\] \[2, 2\] \[1, 1\]}} : tensor<4x2xi7> to tensor<2x2xi7>
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map2, #map3, #map4{{\], iterator}}_types = {{\[}}"parallel", "parallel", "reduction"{{\]}}} ins(%[[Vextracted_slice_0]], %[[Vextracted_slice_1]] : tensor<8x2x!FHE.eint<6>>, tensor<2x2xi7>) outs(%[[Vextracted_slice]] : tensor<8x2x!FHE.eint<6>>) attrs =  {"tile-sizes" = {{\[0, 0, 2\]}}} {
// CHECK-NEXT:       ^bb0(%[[Vin:.*]]: !FHE.eint<6>, %[[Vin_2:.*]]: i7, %[[Vout:.*]]: !FHE.eint<6>):
// CHECK-NEXT:         %[[V9:.*]] = "FHE.mul_eint_int"(%[[Vin]], %[[Vin_2]]) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
// CHECK-NEXT:         %[[V10:.*]] = "FHE.add_eint"(%[[Vout]], %[[V9]]) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
// CHECK-NEXT:         linalg.yield %[[V10]] : !FHE.eint<6>
// CHECK-NEXT:       } -> tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       scf.forall.in_parallel {
// CHECK-NEXT:         tensor.parallel_insert_slice %[[V8]] into %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x!FHE.eint<6>> into tensor<8x2x2x!FHE.eint<6>>
// CHECK-NEXT:       }
// CHECK-NEXT:     }

func.func @tiled_2(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [0,0,2] } : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
  return %0 : tensor<8x2x!FHE.eint<6>>
}

// -----

// CHECK:     %[[V4:.*]] = scf.forall (%[[Varg2:.*]]) in (1) shared_outs(%[[Varg3:.*]] = %[[V3:.*]]) -> (tensor<8x2x1x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x1x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       %[[V6:.*]] = affine.apply #map1(%[[Varg2]])
// CHECK-NEXT:       %[[V7:.*]] = affine.apply #map1(%[[Varg2]])
// CHECK-NEXT:       %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg0:.*]]{{\[0,}} %[[V6]]{{\] \[8, 4\] \[1, 1\]}} : tensor<8x4x!FHE.eint<6>> to tensor<8x4x!FHE.eint<6>>
// CHECK-NEXT:       %[[Vextracted_slice_1:.*]] = tensor.extract_slice %[[Varg1:.*]]{{\[}}%[[V7]], 0{{\] \[4, 2\] \[1, 1\]}} : tensor<4x2xi7> to tensor<4x2xi7>
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map2, #map3, #map4{{\], iterator}}_types = {{\[}}"parallel", "parallel", "reduction"{{\]}}} ins(%[[Vextracted_slice_0]], %[[Vextracted_slice_1]] : tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) outs(%[[Vextracted_slice]] : tensor<8x2x!FHE.eint<6>>) attrs =  {"tile-sizes" = {{\[0, 0, 4\]}}} {
// CHECK-NEXT:       ^bb0(%[[Vin:.*]]: !FHE.eint<6>, %[[Vin_2:.*]]: i7, %[[Vout:.*]]: !FHE.eint<6>):
// CHECK-NEXT:         %[[V9:.*]] = "FHE.mul_eint_int"(%[[Vin]], %[[Vin_2]]) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
// CHECK-NEXT:         %[[V10:.*]] = "FHE.add_eint"(%[[Vout]], %[[V9]]) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
// CHECK-NEXT:         linalg.yield %[[V10]] : !FHE.eint<6>
// CHECK-NEXT:       } -> tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       scf.forall.in_parallel {
// CHECK-NEXT:         tensor.parallel_insert_slice %[[V8]] into %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x!FHE.eint<6>> into tensor<8x2x1x!FHE.eint<6>>
// CHECK-NEXT:       }
// CHECK-NEXT:     }

func.func @tiled_one_big_tile(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [0,0,4] } : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
  return %0 : tensor<8x2x!FHE.eint<6>>
}

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

// -----

// CHECK:     %[[V1:.*]] = scf.forall (%[[Varg2:.*]], %[[Varg3:.*]], %[[Varg4:.*]]) in (1, 1, 2) shared_outs(%[[Varg5:.*]] = %[[V0:.*]]) -> (tensor<2x3x4x!FHE.eint<2>>) {
// CHECK-NEXT:       %[[V2:.*]] = affine.apply #map(%[[Varg2]])
// CHECK-NEXT:       %[[V3:.*]] = affine.apply #map1(%[[Varg3]])
// CHECK-NEXT:       %[[V4:.*]] = affine.apply #map(%[[Varg4]])
// CHECK-NEXT:       %[[V5:.*]] = affine.apply #map(%[[Varg2]])
// CHECK-NEXT:       %[[V6:.*]] = affine.apply #map1(%[[Varg3]])
// CHECK-NEXT:       %[[V7:.*]] = affine.apply #map(%[[Varg4]])
// CHECK-NEXT:       %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[V2]], %[[V3]], %[[V4]]{{\] \[2, 3, 2\] \[1, 1, 1\]}} : tensor<2x3x4x!FHE.eint<2>> to tensor<2x3x2x!FHE.eint<2>>
// CHECK-NEXT:       %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg5]]{{\[}}%[[V5]], %[[V6]], %[[V7]]{{\] \[2, 3, 2\] \[1, 1, 1\]}} : tensor<2x3x4x!FHE.eint<2>> to tensor<2x3x2x!FHE.eint<2>>
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map2, #map2{{\], iterator}}_types = {{\[}}"parallel", "parallel", "parallel"{{\]}}} ins(%[[Vextracted_slice]] : tensor<2x3x2x!FHE.eint<2>>) outs(%[[Vextracted_slice_0]] : tensor<2x3x2x!FHE.eint<2>>) attrs =  {"tile-sizes" = {{\[2, 3, 2\]}}} {
// CHECK-NEXT:       ^bb0(%[[Vin:.*]]: !FHE.eint<2>, %[[Vout:.*]]: !FHE.eint<2>):
// CHECK-NEXT:         %[[V12:.*]] = "FHE.apply_lookup_table"(%[[Vin]], %[[Varg1]]) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
// CHECK-NEXT:         linalg.yield %[[V12]] : !FHE.eint<2>
// CHECK-NEXT:       } -> tensor<2x3x2x!FHE.eint<2>>
// CHECK-NEXT:       %[[V9:.*]] = affine.apply #map(%[[Varg2]])
// CHECK-NEXT:       %[[V10:.*]] = affine.apply #map1(%[[Varg3]])
// CHECK-NEXT:       %[[V11:.*]] = affine.apply #map(%[[Varg4]])
// CHECK-NEXT:       scf.forall.in_parallel {
// CHECK-NEXT:         tensor.parallel_insert_slice %[[V8]] into %[[Varg5]]{{\[}}%[[V9]], %[[V10]], %[[V11]]{{\] \[2, 3, 2\] \[1, 1, 1\]}} : tensor<2x3x2x!FHE.eint<2>> into tensor<2x3x4x!FHE.eint<2>>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
func.func @apply_lookup_table(%arg0: tensor<2x3x4x!FHE.eint<2>>, %arg1: tensor<4xi64>) -> tensor<2x3x4x!FHE.eint<2>> {
  %1 = "FHELinalg.apply_lookup_table"(%arg0, %arg1) { "tile-sizes" = [2,3,2] } : (tensor<2x3x4x!FHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}

// -----

// CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[map:.*]], #[[map2:.*]], #[[map3:.*]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[extracted_slice:.*]], %[[extracted_slice_0:.*]] : tensor<2x3x2x!FHE.eint<7>>, tensor<2x3x2xindex>) outs(%[[extracted_slice_1:.*]] : tensor<2x3x2x!FHE.eint<7>>) attrs =  {"tile-sizes" = [2, 3, 2]}
func.func @apply_mapped_lookup_table(
  %input: tensor<2x3x4x!FHE.eint<7>>,
  %luts: tensor<10x128xi64>,
  %map: tensor<2x3x4xindex>
) -> tensor<2x3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.apply_mapped_lookup_table"(%input, %luts, %map) { "tile-sizes" = [2,3,2] } : (tensor<2x3x4x!FHE.eint<7>>, tensor<10x128xi64>, tensor<2x3x4xindex>) -> (tensor<2x3x4x!FHE.eint<7>>)
  return %0: tensor<2x3x4x!FHE.eint<7>>
}

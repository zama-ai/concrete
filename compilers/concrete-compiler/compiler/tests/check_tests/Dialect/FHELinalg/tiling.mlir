// RUN: concretecompiler --action=dump-fhe-no-linalg %s 2>&1 --optimizer-strategy=dag-mono --split-input-file | FileCheck %s

// CHECK:     %[[V4:.*]] = scf.forall (%[[Varg2:.*]]) in (2) shared_outs(%[[Varg3:.*]] = %[[V3:.*]]) -> (tensor<8x2x2x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x2x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       %[[V6:.*]] = affine.apply #map1(%[[Varg2]])
// CHECK-NEXT:       %[[V7:.*]] = affine.apply #map1(%[[Varg2]])
// CHECK-NEXT:       %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg0:.*]]{{\[0,}} %[[V6]]{{\] \[8, 2\] \[1, 1\]}} : tensor<8x4x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       %[[Vextracted_slice_1:.*]] = tensor.extract_slice %[[Varg1:.*]]{{\[}}%[[V7]], 0{{\] \[2, 2\] \[1, 1\]}} : tensor<4x2xi7> to tensor<2x2xi7>
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map2, #map3, #map4{{\], iterator}}_types = {{\[}}"parallel", "parallel", "reduction"{{\]}}} ins(%[[Vextracted_slice_0]], %[[Vextracted_slice_1]] : tensor<8x2x!FHE.eint<6>>, tensor<2x2xi7>) outs(%[[Vextracted_slice]] : tensor<8x2x!FHE.eint<6>>) {
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
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map2, #map3, #map4{{\], iterator}}_types = {{\[}}"parallel", "parallel", "reduction"{{\]}}} ins(%[[Vextracted_slice_0]], %[[Vextracted_slice_1]] : tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) outs(%[[Vextracted_slice]] : tensor<8x2x!FHE.eint<6>>) {
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
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map2, #map2{{\], iterator}}_types = {{\[}}"parallel", "parallel", "parallel"{{\]}}} ins(%[[Vextracted_slice]] : tensor<2x3x2x!FHE.eint<2>>) outs(%[[Vextracted_slice_0]] : tensor<2x3x2x!FHE.eint<2>>) {
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

// CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[map:.*]], #[[map2:.*]], #[[map3:.*]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[extracted_slice:.*]], %[[extracted_slice_0:.*]] : tensor<2x3x2x!FHE.eint<7>>, tensor<2x3x2xindex>) outs(%[[extracted_slice_1:.*]] : tensor<2x3x2x!FHE.eint<7>>)
func.func @apply_mapped_lookup_table(
  %input: tensor<2x3x4x!FHE.eint<7>>,
  %luts: tensor<10x128xi64>,
  %map: tensor<2x3x4xindex>
) -> tensor<2x3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.apply_mapped_lookup_table"(%input, %luts, %map) { "tile-sizes" = [2,3,2] } : (tensor<2x3x4x!FHE.eint<7>>, tensor<10x128xi64>, tensor<2x3x4xindex>) -> (tensor<2x3x4x!FHE.eint<7>>)
  return %0: tensor<2x3x4x!FHE.eint<7>>
}

// -----

// CHECK: %[[res:.*]] = linalg.generic {indexing_maps = [#[[map:.*]], #[[map2:.*]]], iterator_types = ["parallel", "parallel"]} ins(%[[extracted_slice:.*]] : tensor<3x3x!FHE.eint<2>>) outs(%[[extracted_slice_0:.*]] : tensor<3x3x!FHE.eint<2>>) {
func.func @main(%arg0: tensor<3x3x!FHE.eint<2>>, %arg1: tensor<3x3x4xi8>) -> tensor<3x3x!FHE.eint<2>> {
  %1 = "FHELinalg.apply_multi_lookup_table"(%arg0, %arg1) { "tile-sizes" = [3, 3] }: (tensor<3x3x!FHE.eint<2>>, tensor<3x3x4xi8>) -> tensor<3x3x!FHE.eint<2>>
  return %1: tensor<3x3x!FHE.eint<2>>
}

// -----

// CHECK:      %[[V4:.*]] = scf.forall (%[[Varg1]], %[[Varg2]], %[[Varg3]]) in (1, 1, 2) shared_outs(%[[Varg4:.*]] = %[[V3]]) -> (tensor<3x8x2x!FHE.eint<7>>) {
// CHECK-NEXT:   %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg4]]{{\[0, 0,}} %[[Varg1]]{{\] \[3, 8, 1\] \[1, 1, 1\]}} : tensor<3x8x2x!FHE.eint<7>> to tensor<3x8x!FHE.eint<7>>
// CHECK-NEXT:   %[[V6:.*]] = affine.apply #map1(%[[Varg1]])
// CHECK-NEXT:   %[[V7:.*]] = affine.apply #map2(%[[Varg2]])
// CHECK-NEXT:   %[[V8:.*]] = affine.apply #map3(%[[Varg3]])
// CHECK-NEXT:   %[[V9:.*]] = affine.apply #map1(%[[Varg1]])
// CHECK-NEXT:   %[[V10:.*]] = affine.apply #map2(%[[Varg2]])
// CHECK-NEXT:   %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[V6]], %[[V7]], %[[V8]]{{\] \[3, 8, 2\] \[1, 1, 1\]}} : tensor<3x8x4x!FHE.eint<7>> to tensor<3x8x2x!FHE.eint<7>>
// CHECK-NEXT:   %[[Vextracted_slice_1:.*]] = tensor.extract_slice %[[Vextracted_slice]]{{\[}}%[[V9]], %[[V10]]{{\] \[3, 8\] \[1, 1\]}} : tensor<3x8x!FHE.eint<7>> to tensor<3x8x!FHE.eint<7>>
// CHECK-NEXT:   %[[V11:.*]] = linalg.generic {indexing_maps = {{\[}}#map, #map4{{\], iterator}}_types = {{\[}}"parallel", "parallel", "reduction"{{\]}}} ins(%[[Vextracted_slice_0]] : tensor<3x8x2x!FHE.eint<7>>) outs(%[[Vextracted_slice_1]] : tensor<3x8x!FHE.eint<7>>) {
// CHECK-NEXT:   ^bb0(%[[Vin:.*]]: !FHE.eint<7>, %[[Vout:.*]]: !FHE.eint<7>):
// CHECK-NEXT:     %[[V14:.*]] = "FHE.add_eint"(%[[Vin]], %[[Vout]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:     linalg.yield %[[V14]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<3x8x!FHE.eint<7>>
// CHECK-NEXT:   %[[V12:.*]] = affine.apply #map1(%[[Varg1]])
// CHECK-NEXT:   %[[V13:.*]] = affine.apply #map2(%[[Varg2]])
// CHECK-NEXT:   scf.forall.in_parallel {
// CHECK-NEXT:     tensor.parallel_insert_slice %[[V11]] into %[[Varg4]]{{\[}}%[[V12]], %[[V13]], %[[Varg1]]{{\] \[3, 8, 1\] \[1, 1, 1\]}} : tensor<3x8x!FHE.eint<7>> into tensor<3x8x2x!FHE.eint<7>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @main(%arg0: tensor<3x8x4x!FHE.eint<7>>) -> tensor<3x8x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [2], "tile-sizes" = [3, 8, 2] } : (tensor<3x8x4x!FHE.eint<7>>) -> tensor<3x8x!FHE.eint<7>>
  return %0 : tensor<3x8x!FHE.eint<7>>
}

// -----

// CHECK:      #map = affine_map<(d0) -> (d0 * -3 + 2, 3)>
// CHECK-NEXT: #map1 = affine_map<(d0) -> (d0 * -8 + 10, 8)>
// CHECK-NEXT: #map2 = affine_map<(d0) -> (d0 * 3)>
// CHECK-NEXT: #map3 = affine_map<(d0) -> (d0 * 8)>
// CHECK-NEXT: #map4 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: #map5 = affine_map<(d0, d1) -> (d1, d0)>
// CHECK-NEXT: module {
// CHECK-NEXT:   func.func @transpose_eint_2D(%[[Varg0:.*]]: tensor<2x10x!FHE.eint<6>>) -> tensor<10x2x!FHE.eint<6>> {
// CHECK-NEXT:     %[[V0:.*]] = "FHE.zero_tensor"() : () -> tensor<10x2x!FHE.eint<6>>
// CHECK-NEXT:     %[[V1:.*]] = scf.forall (%[[Varg1]], %[[Varg2]]) in (1, 2) shared_outs(%[[Varg3:.*]] = %[[V0]]) -> (tensor<10x2x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[V2:.*]] = affine.min #map(%[[Varg1]])
// CHECK-NEXT:       %[[V3:.*]] = affine.min #map1(%[[Varg2]])
// CHECK-NEXT:       %[[V4:.*]] = affine.apply #map2(%[[Varg1]])
// CHECK-NEXT:       %[[V5:.*]] = affine.apply #map3(%[[Varg2]])
// CHECK-NEXT:       %[[V6:.*]] = affine.apply #map3(%[[Varg2]])
// CHECK-NEXT:       %[[V7:.*]] = affine.apply #map2(%[[Varg1]])
// CHECK-NEXT:       %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg0]]{{\[}}%[[V4]], %[[V5]]{{\] \[}}%[[V2]], %[[V3]]{{\] \[1, 1\]}} : tensor<2x10x!FHE.eint<6>> to tensor<?x?x!FHE.eint<6>>
// CHECK-NEXT:       %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg3]]{{\[}}%[[V6]], %[[V7]]{{\] \[}}%[[V3]], %[[V2]]{{\] \[1, 1\]}} : tensor<10x2x!FHE.eint<6>> to tensor<?x?x!FHE.eint<6>>
// CHECK-NEXT:       %[[V8:.*]] = linalg.generic {indexing_maps = {{\[}}#map4, #map5{{\], iterator}}_types = {{\[}}"parallel", "parallel"{{\]}}} ins(%[[Vextracted_slice]] : tensor<?x?x!FHE.eint<6>>) outs(%[[Vextracted_slice_0]] : tensor<?x?x!FHE.eint<6>>) {
// CHECK-NEXT:       ^bb0(%[[Vin:.*]]: !FHE.eint<6>, %[[Vout:.*]]: !FHE.eint<6>):
// CHECK-NEXT:         linalg.yield %[[Vin]] : !FHE.eint<6>
// CHECK-NEXT:       } -> tensor<?x?x!FHE.eint<6>>
// CHECK-NEXT:       %[[V9:.*]] = affine.apply #map3(%[[Varg2]])
// CHECK-NEXT:       %[[V10:.*]] = affine.apply #map2(%[[Varg1]])
// CHECK-NEXT:       scf.forall.in_parallel {
// CHECK-NEXT:         tensor.parallel_insert_slice %[[V8]] into %[[Varg3]]{{\[}}%[[V9]], %[[V10]]{{\] \[}}%[[V3]], %[[V2]]{{\] \[1, 1\]}} : tensor<?x?x!FHE.eint<6>> into tensor<10x2x!FHE.eint<6>>
// CHECK-NEXT:       }
// CHECK-NEXT:     }
// CHECK-NEXT:     return %[[V1]] : tensor<10x2x!FHE.eint<6>>
// CHECK-NEXT:   }
// CHECK-NEXT: }

func.func @transpose_eint_2D(%arg0: tensor<2x10x!FHE.eint<6>>) -> tensor<10x2x!FHE.eint<6>> {
  %c = "FHELinalg.transpose"(%arg0) { "tile-sizes" = [3, 8, 2] } : (tensor<2x10x!FHE.eint<6>>) -> tensor<10x2x!FHE.eint<6>>
  return %c : tensor<10x2x!FHE.eint<6>>
}

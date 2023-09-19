// RUN: concretecompiler --action=dump-fhe-no-linalg %s 2>&1 --split-input-file | FileCheck %s

// CHECK: %[[V4:.*]] = scf.forall (%[[Varg2:.*]]) in (2) shared_outs(%[[Varg3:.*]] = %[[V3:.*]]) -> (tensor<8x2x2x!FHE.eint<6>>) {
// CHECK-NEXT:    %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x2x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:    %[[V6:.*]] = affine.apply {{\#}}map{{\(\)}}{{\[}}%[[Varg2]], %[[Vc2:.*]]{{\]}}
// CHECK-NEXT:    %[[V7:.*]] = affine.apply {{\#}}map1{{\(\)}}{{\[}}%[[V6]], %[[Vc0:.*]]{{\]}}
// CHECK-NEXT:    %[[V8:.*]] = scf.for %[[Varg4:.*]] = %[[V7]] to %[[Vc4:.*]] step %[[Vc4]] iter_args(%[[Varg5:.*]] = %[[Vextracted_slice]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:      %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg0:.*]]{{\[0,}} %[[Varg4]]{{\] \[8, 2\] \[1, 1\]}} : tensor<8x4x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:      %[[Vextracted_slice_1:.*]] = tensor.extract_slice %[[Varg1:.*]]{{\[}}%[[Varg4]], 0{{\] \[2, 2\] \[1, 1\]}} : tensor<4x2xi7> to tensor<2x2xi7>
// CHECK-NEXT:      %[[V9:.*]] = scf.for %[[Varg6:.*]] = %[[Vc0]] to %[[Vc8:.*]] step %[[Vc1:.*]] iter_args(%[[Varg7:.*]] = %[[Varg5]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:        %[[V10:.*]] = scf.for %[[Varg8:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg9:.*]] = %[[Varg7]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:          %[[V11:.*]] = scf.for %[[Varg10:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg11:.*]] = %[[Varg9]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:            %[[Vextracted:.*]] = tensor.extract %[[Vextracted_slice_0]]{{\[}}%[[Varg6]], %[[Varg10]]{{\]}} : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:            %[[Vextracted_2:.*]] = tensor.extract %[[Vextracted_slice_1]]{{\[}}%[[Varg10]], %[[Varg8]]{{\]}} : tensor<2x2xi7>
// CHECK-NEXT:            %[[Vextracted_3:.*]] = tensor.extract %[[Varg11]]{{\[}}%[[Varg6]], %[[Varg8]]{{\]}} : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:            %[[V12:.*]] = "FHE.mul_eint_int"(%[[Vextracted]], %[[Vextracted_2]]) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
// CHECK-NEXT:            %[[V13:.*]] = "FHE.add_eint"(%[[Vextracted_3]], %[[V12]]) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
// CHECK-NEXT:            %[[Vinserted:.*]] = tensor.insert %[[V13]] into %[[Varg11]]{{\[}}%[[Varg6]], %[[Varg8]]{{\]}} : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:            scf.yield %[[Vinserted]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:          }
// CHECK-NEXT:          scf.yield %[[V11]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[V10]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V9]] : tensor<8x2x!FHE.eint<6>> 
// CHECK-NEXT:    }
// CHECK-NEXT:    scf.forall.in_parallel {
// CHECK-NEXT:      tensor.parallel_insert_slice %[[V8]] into %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x!FHE.eint<6>> into tensor<8x2x2x!FHE.eint<6>>
// CHECK-NEXT:    }
// CHECK-NEXT:  }
func.func @tiled_2(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [0,0,2] } : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
  return %0 : tensor<8x2x!FHE.eint<6>>
}

// -----

// CHECK:  %[[V4:.*]] = scf.forall (%[[Varg2:.*]]) in (1) shared_outs(%[[Varg3:.*]] = %[[V3:.*]]) -> (tensor<8x2x1x!FHE.eint<6>>) {
// CHECK-NEXT:   %[[Vextracted_slice:.*]] = tensor.extract_slice %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x1x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:   %[[V6:.*]] = affine.apply {{\#}}map{{\(\)}}{{\[}}%[[Varg2]], %[[Vc4:.*]]{{\]}}
// CHECK-NEXT:   %[[V7:.*]] = affine.apply {{\#}}map1{{\(\)}}{{\[}}%[[V6]], %[[Vc0:.*]]{{\]}}
// CHECK-NEXT:   %[[V8:.*]] = scf.for %[[Varg4:.*]] = %[[V7]] to %[[Vc4]] step %[[Vc4]] iter_args(%[[Varg5:.*]] = %[[Vextracted_slice]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[Vextracted_slice_0:.*]] = tensor.extract_slice %[[Varg0:.*]]{{\[0,}} %[[Varg4]]{{\] \[8, 4\] \[1, 1\]}} : tensor<8x4x!FHE.eint<6>> to tensor<8x4x!FHE.eint<6>>
// CHECK-NEXT:     %[[Vextracted_slice_1:.*]] = tensor.extract_slice %[[Varg1:.*]]{{\[}}%[[Varg4]], 0{{\] \[4, 2\] \[1, 1\]}} : tensor<4x2xi7> to tensor<4x2xi7>
// CHECK-NEXT:     %[[V9:.*]] = scf.for %[[Varg6:.*]] = %[[Vc0]] to %[[Vc8:.*]] step %[[Vc1]] iter_args(%[[Varg7:.*]] = %[[Varg5]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[V10:.*]] = scf.for %[[Varg8:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc1]] iter_args(%[[Varg9:.*]] = %[[Varg7]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:         %[[V11:.*]] = scf.for %[[Varg10:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc1]] iter_args(%[[Varg11:.*]] = %[[Varg9]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:           %[[Vextracted:.*]] = tensor.extract %[[Vextracted_slice_0]]{{\[}}%[[Varg6]], %[[Varg10]]{{\]}} : tensor<8x4x!FHE.eint<6>>
// CHECK-NEXT:           %[[Vextracted_2:.*]] = tensor.extract %[[Vextracted_slice_1]]{{\[}}%[[Varg10]], %[[Varg8]]{{\]}} : tensor<4x2xi7>
// CHECK-NEXT:           %[[Vextracted_3:.*]] = tensor.extract %[[Varg11]]{{\[}}%[[Varg6]], %[[Varg8]]{{\]}} : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:           %[[V12:.*]] = "FHE.mul_eint_int"(%[[Vextracted]], %[[Vextracted_2]]) : (!FHE.eint<6>, i7) -> !FHE.eint<6>
// CHECK-NEXT:           %[[V13:.*]] = "FHE.add_eint"(%[[Vextracted_3]], %[[V12]]) : (!FHE.eint<6>, !FHE.eint<6>) -> !FHE.eint<6>
// CHECK-NEXT:           %[[Vinserted:.*]] = tensor.insert %[[V13]] into %[[Varg11]]{{\[}}%[[Varg6]], %[[Varg8]]{{\]}} : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:           scf.yield %[[Vinserted]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:         }
// CHECK-NEXT:         scf.yield %[[V11]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[V10]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[V9]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:   }
// CHECK-NEXT:   scf.forall.in_parallel {
// CHECK-NEXT:     tensor.parallel_insert_slice %[[V8]] into %[[Varg3]]{{\[0, 0,}} %[[Varg2]]{{\] \[8, 2, 1\] \[1, 1, 1\]}} : tensor<8x2x!FHE.eint<6>> into tensor<8x2x1x!FHE.eint<6>>
// CHECK-NEXT:   }
// CHECK-NEXT: }
func.func @tiled_one_big_tile(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [0,0,4] } : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
  return %0 : tensor<8x2x!FHE.eint<6>>
}

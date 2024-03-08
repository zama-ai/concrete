// RUN: concretecompiler --action=dump-fhe %s 2>&1 --optimizer-strategy=dag-mono --split-input-file | FileCheck %s

// CHECK:      func.func @tiled_2x2(%[[Varg0:.*]]: tensor<8x4x!FHE.eint<6>>, %[[Varg1:.*]]: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
// CHECK-NEXT:    %[[Vc2:.*]] = arith.constant 2 : index
// CHECK-NEXT:    %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:    %[[Vc8:.*]] = arith.constant 8 : index
// CHECK-NEXT:    %[[Vc4:.*]] = arith.constant 4 : index
// CHECK-NEXT:    %[[V0:.*]] = "FHE.zero_tensor"() : () -> tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:    %[[V1:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc8]] step %[[Vc2]] iter_args(%[[Varg3:.*]] = %[[V0]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:      %[[V2:.*]] = scf.for %[[Varg4:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc2]] iter_args(%[[Varg5:.*]] = %[[Varg3]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:        %[[V3:.*]] = scf.for %[[Varg6:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc2]] iter_args(%[[Varg7:.*]] = %[[Varg5]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:          %[[V4:.*]] = tensor.extract_slice %[[Varg0]][%[[Varg2]], %[[Varg4]]] [2, 2] [1, 1] : tensor<8x4x!FHE.eint<6>> to tensor<2x2x!FHE.eint<6>>
// CHECK-NEXT:          %[[V5:.*]] = tensor.extract_slice %[[Varg1]][%[[Varg4]], %[[Varg6]]] [2, 2] [1, 1] : tensor<4x2xi7> to tensor<2x2xi7>
// CHECK-NEXT:          %[[V6:.*]] = tensor.extract_slice %[[Varg7]][%[[Varg2]], %[[Varg6]]] [2, 2] [1, 1] : tensor<8x2x!FHE.eint<6>> to tensor<2x2x!FHE.eint<6>>
// CHECK-NEXT:          %[[V7:.*]] = "FHELinalg.matmul_eint_int"(%[[V4]], %[[V5]]) : (tensor<2x2x!FHE.eint<6>>, tensor<2x2xi7>) -> tensor<2x2x!FHE.eint<6>>
// CHECK-NEXT:          %[[V8:.*]] = "FHELinalg.add_eint"(%[[V6]], %[[V7]]) : (tensor<2x2x!FHE.eint<6>>, tensor<2x2x!FHE.eint<6>>) -> tensor<2x2x!FHE.eint<6>>
// CHECK-NEXT:          %[[V9:.*]] = tensor.insert_slice %[[V8]] into %[[Varg7]][%[[Varg2]], %[[Varg6]]] [2, 2] [1, 1] : tensor<2x2x!FHE.eint<6>> into tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:          scf.yield %[[V9]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:        }
// CHECK-NEXT:        scf.yield %[[V3]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:      }
// CHECK-NEXT:      scf.yield %[[V2]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:    }
// CHECK-NEXT:    return %[[V1]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @tiled_2x2(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [2,2,2] } : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
  return %0 : tensor<8x2x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @tiled_one_big_tile(%[[Varg0:.*]]: tensor<8x4x!FHE.eint<6>>, %[[Varg1:.*]]: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
// CHECK-NEXT:   %[[Vc8:.*]] = arith.constant 8 : index
// CHECK-NEXT:   %[[Vc4:.*]] = arith.constant 4 : index
// CHECK-NEXT:   %[[Vc2:.*]] = arith.constant 2 : index
// CHECK-NEXT:   %[[Vc0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[V0:.*]] = "FHE.zero_tensor"() : () -> tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:   %[[V1:.*]] = scf.for %[[Varg2:.*]] = %[[Vc0]] to %[[Vc8]] step %[[Vc8]] iter_args(%[[Varg3:.*]] = %[[V0]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[V2:.*]] = scf.for %[[Varg4:.*]] = %[[Vc0]] to %[[Vc4]] step %[[Vc4]] iter_args(%[[Varg5:.*]] = %[[Varg3]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:       %[[V3:.*]] = scf.for %[[Varg6:.*]] = %[[Vc0]] to %[[Vc2]] step %[[Vc2]] iter_args(%[[Varg7:.*]] = %[[Varg5]]) -> (tensor<8x2x!FHE.eint<6>>) {
// CHECK-NEXT:         %[[V4:.*]] = tensor.extract_slice %[[Varg0]][%[[Varg2]], %[[Varg4]]] [8, 4] [1, 1] : tensor<8x4x!FHE.eint<6>> to tensor<8x4x!FHE.eint<6>>
// CHECK-NEXT:         %[[V5:.*]] = tensor.extract_slice %[[Varg1]][%[[Varg4]], %[[Varg6]]] [4, 2] [1, 1] : tensor<4x2xi7> to tensor<4x2xi7>
// CHECK-NEXT:         %[[V6:.*]] = tensor.extract_slice %[[Varg7]][%[[Varg2]], %[[Varg6]]] [8, 2] [1, 1] : tensor<8x2x!FHE.eint<6>> to tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:         %[[V7:.*]] = "FHELinalg.matmul_eint_int"(%[[V4]], %[[V5]]) : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:         %[[V8:.*]] = "FHELinalg.add_eint"(%[[V6]], %[[V7]]) : (tensor<8x2x!FHE.eint<6>>, tensor<8x2x!FHE.eint<6>>) -> tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:         %[[V9:.*]] = tensor.insert_slice %[[V8]] into %[[Varg7]][%[[Varg2]], %[[Varg6]]] [8, 2] [1, 1] : tensor<8x2x!FHE.eint<6>> into tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:         scf.yield %[[V9]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:       }
// CHECK-NEXT:       scf.yield %[[V3]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:     scf.yield %[[V2]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[V1]] : tensor<8x2x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @tiled_one_big_tile(%a: tensor<8x4x!FHE.eint<6>>, %b: tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>> {
  %0 = "FHELinalg.matmul_eint_int"(%a, %b) { "tile-sizes" = [8,4,2] } : (tensor<8x4x!FHE.eint<6>>, tensor<4x2xi7>) -> tensor<8x2x!FHE.eint<6>>
  return %0 : tensor<8x2x!FHE.eint<6>>
}


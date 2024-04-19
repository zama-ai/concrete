// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @from_1d_to_1d(%[[input:.*]]: tensor<5x!FHE.eint<6>>, %[[indices:.*]]: tensor<3xindex>) -> tensor<3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index):
// CHECK-NEXT:     %[[index:.*]] = tensor.extract %[[indices]][%[[i]]] : tensor<3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index]]] : tensor<5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_to_1d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_to_2d(%[[input:.*]]: tensor<5x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index):
// CHECK-NEXT:     %[[index:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]]] : tensor<2x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index]]] : tensor<5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_to_2d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_to_3d(%[[input:.*]]: tensor<5x!FHE.eint<6>>, %[[indices:.*]]: tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index, %[[k:.*]]: index):
// CHECK-NEXT:     %[[index:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]]] : tensor<4x2x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index]]] : tensor<5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<4x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<4x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_to_3d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>>
  return %output : tensor<4x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_1d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[c0]]] : tensor<3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[c1]]] : tensor<3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]]] : tensor<4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_1d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_2d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c0]]] : tensor<2x3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c1]]] : tensor<2x3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]]] : tensor<4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_2d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_3d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index, %[[k:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c0]]] : tensor<6x2x3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c1]]] : tensor<6x2x3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]]] : tensor<4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_3d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>>
  return %output : tensor<6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_4d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index, %[[k:.*]]: index, %[[l:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c0]]] : tensor<5x6x2x3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c1]]] : tensor<5x6x2x3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]]] : tensor<4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_4d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>>
  return %output : tensor<5x6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_1d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[c0]]] : tensor<3x3xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[c1]]] : tensor<3x3xindex>
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[index2:.*]] = tensor.extract %[[indices]][%[[i]], %[[c2]]] : tensor<3x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]], %[[index2]]] : tensor<2x4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_1d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_2d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c0]]] : tensor<2x3x3xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c1]]] : tensor<2x3x3xindex>
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[index2:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c2]]] : tensor<2x3x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]], %[[index2]]] : tensor<2x4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_2d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_3d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index, %[[k:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c0]]] : tensor<6x2x3x3xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c1]]] : tensor<6x2x3x3xindex>
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[index2:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c2]]] : tensor<6x2x3x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]], %[[index2]]] : tensor<2x4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_3d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>>
  return %output : tensor<6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_4d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<5x6x2x3x3xindex>) -> tensor<5x6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index, %[[k:.*]]: index, %[[l:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c0]]] : tensor<5x6x2x3x3xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c1]]] : tensor<5x6x2x3x3xindex>
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[index2:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c2]]] : tensor<5x6x2x3x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]], %[[index2]]] : tensor<2x4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_4d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<5x6x2x3x3xindex>) -> tensor<5x6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<5x6x2x3x3xindex>) -> tensor<5x6x2x3x!FHE.eint<6>>
  return %output : tensor<5x6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_5d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<4x5x6x2x3x3xindex>) -> tensor<4x5x6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = tensor.generate  {
// CHECK-NEXT:   ^bb0(%[[i:.*]]: index, %[[j:.*]]: index, %[[k:.*]]: index, %[[l:.*]]: index, %[[m:.*]]: index):
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[m]], %[[c0]]] : tensor<4x5x6x2x3x3xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[m]], %[[c1]]] : tensor<4x5x6x2x3x3xindex>
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[index2:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[m]], %[[c2]]] : tensor<4x5x6x2x3x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[input]][%[[index0]], %[[index1]], %[[index2]]] : tensor<2x4x5x!FHE.eint<6>>
// CHECK-NEXT:     tensor.yield %[[element]] : !FHE.eint<6>
// CHECK-NEXT:   } : tensor<4x5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<4x5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_5d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<4x5x6x2x3x3xindex>) -> tensor<4x5x6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<4x5x6x2x3x3xindex>) -> tensor<4x5x6x2x3x!FHE.eint<6>>
  return %output : tensor<4x5x6x2x3x!FHE.eint<6>>
}

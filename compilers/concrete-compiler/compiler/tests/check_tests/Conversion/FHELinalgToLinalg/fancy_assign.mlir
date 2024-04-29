// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @from_1d_into_1d(%[[input:.*]]: tensor<25x!FHE.eint<6>>, %[[indices:.*]]: tensor<3xindex>, %[[values:.*]]: tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]]) in (3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<25x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[index:.*]] = tensor.extract %[[indices]][%[[i]]] : tensor<3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index]]] [1] [1] : tensor<1x!FHE.eint<6>> into tensor<25x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<25x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_into_1d(%input: tensor<25x!FHE.eint<6>>, %indices: tensor<3xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<25x!FHE.eint<6>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
  return %output : tensor<25x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_into_1d(%[[input:.*]]: tensor<25x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3xindex>, %[[values:.*]]: tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]], %[[j:.*]]) in (2, 3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<25x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[index:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]]] : tensor<2x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]], %[[j]]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index]]] [1] [1] : tensor<1x!FHE.eint<6>> into tensor<25x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<25x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_into_1d(%input: tensor<25x!FHE.eint<6>>, %indices: tensor<2x3xindex>, %values: tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<25x!FHE.eint<6>>, tensor<2x3xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
  return %output : tensor<25x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_into_1d(%[[input:.*]]: tensor<25x!FHE.eint<6>>, %[[indices:.*]]: tensor<4x2x3xindex>, %[[values:.*]]: tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]], %[[j:.*]], %[[k:.*]]) in (4, 2, 3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<25x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[index:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]]] : tensor<4x2x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]], %[[j]], %[[k]]] : tensor<4x2x3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index]]] [1] [1] : tensor<1x!FHE.eint<6>> into tensor<25x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<25x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_into_1d(%input: tensor<25x!FHE.eint<6>>, %indices: tensor<4x2x3xindex>, %values: tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<25x!FHE.eint<6>>, tensor<4x2x3xindex>, tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
  return %output : tensor<25x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_into_2d(%[[input:.*]]: tensor<5x10x!FHE.eint<6>>, %[[indices:.*]]: tensor<3x2xindex>, %[[values:.*]]: tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]]) in (3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<5x10x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[c0]]] : tensor<3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[c1]]] : tensor<3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index0]], %[[index1]]] [1, 1] [1, 1] : tensor<1x!FHE.eint<6>> into tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_into_2d(%input: tensor<5x10x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x10x!FHE.eint<6>>, tensor<3x2xindex>, tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
  return %output : tensor<5x10x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_into_2d(%[[input:.*]]: tensor<5x10x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3x2xindex>, %[[values:.*]]: tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]], %[[j:.*]]) in (2, 3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<5x10x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c0]]] : tensor<2x3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[c1]]] : tensor<2x3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]], %[[j]]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index0]], %[[index1]]] [1, 1] [1, 1] : tensor<1x!FHE.eint<6>> into tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_into_2d(%input: tensor<5x10x!FHE.eint<6>>, %indices: tensor<2x3x2xindex>, %values: tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x10x!FHE.eint<6>>, tensor<2x3x2xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
  return %output : tensor<5x10x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_into_2d(%[[input:.*]]: tensor<5x10x!FHE.eint<6>>, %[[indices:.*]]: tensor<6x2x3x2xindex>, %[[values:.*]]: tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]], %[[j:.*]], %[[k:.*]]) in (6, 2, 3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<5x10x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c0]]] : tensor<6x2x3x2xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[c1]]] : tensor<6x2x3x2xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]], %[[j]], %[[k]]] : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index0]], %[[index1]]] [1, 1] [1, 1] : tensor<1x!FHE.eint<6>> into tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_into_2d(%input: tensor<5x10x!FHE.eint<6>>, %indices: tensor<6x2x3x2xindex>, %values: tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x10x!FHE.eint<6>>, tensor<6x2x3x2xindex>, tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
  return %output : tensor<5x10x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_4d_into_3d(%[[input:.*]]: tensor<20x5x2x!FHE.eint<6>>, %[[indices:.*]]: tensor<5x6x2x3x3xindex>, %[[values:.*]]: tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = scf.forall (%[[i:.*]], %[[j:.*]], %[[k:.*]], %[[l:.*]]) in (5, 6, 2, 3) shared_outs(%[[result:.*]] = %[[input]]) -> (tensor<20x5x2x!FHE.eint<6>>) {
// CHECK-NEXT:     %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:     %[[index0:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c0]]] : tensor<5x6x2x3x3xindex>
// CHECK-NEXT:     %[[c1:.*]] = arith.constant 1 : index
// CHECK-NEXT:     %[[index1:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c1]]] : tensor<5x6x2x3x3xindex>
// CHECK-NEXT:     %[[c2:.*]] = arith.constant 2 : index
// CHECK-NEXT:     %[[index2:.*]] = tensor.extract %[[indices]][%[[i]], %[[j]], %[[k]], %[[l]], %[[c2]]] : tensor<5x6x2x3x3xindex>
// CHECK-NEXT:     %[[element:.*]] = tensor.extract %[[values]][%[[i]], %[[j]], %[[k]], %[[l]]] : tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:     %[[element_slice:.*]] = tensor.from_elements %[[element]] : tensor<1x!FHE.eint<6>>
// CHECK-NEXT:     scf.forall.in_parallel {
// CHECK-NEXT:       tensor.parallel_insert_slice %[[element_slice]] into %[[result]][%[[index0]], %[[index1]], %[[index2]]] [1, 1, 1] [1, 1, 1] : tensor<1x!FHE.eint<6>> into tensor<20x5x2x!FHE.eint<6>>
// CHECK-NEXT:     }
// CHECK-NEXT:   }
// CHECK-NEXT:   return %[[output]] : tensor<20x5x2x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_4d_into_3d(%input: tensor<20x5x2x!FHE.eint<6>>, %indices: tensor<5x6x2x3x3xindex>, %values: tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<20x5x2x!FHE.eint<6>>, tensor<5x6x2x3x3xindex>, tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>>
  return %output : tensor<20x5x2x!FHE.eint<6>>
}

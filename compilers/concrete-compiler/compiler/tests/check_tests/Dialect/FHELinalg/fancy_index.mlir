// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @from_1d_to_1d(%[[input:.*]]: tensor<5x!FHE.eint<6>>, %[[indices:.*]]: tensor<3xindex>) -> tensor<3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_to_1d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<3xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_to_2d(%[[input:.*]]: tensor<5x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<5x!FHE.eint<6>>, tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_to_2d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<2x3xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_to_3d(%[[input:.*]]: tensor<5x!FHE.eint<6>>, %[[indices:.*]]: tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<5x!FHE.eint<6>>, tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<4x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_to_3d(%input: tensor<5x!FHE.eint<6>>, %indices: tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<5x!FHE.eint<6>>, tensor<4x2x3xindex>) -> tensor<4x2x3x!FHE.eint<6>>
  return %output : tensor<4x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_1d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<4x5x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_1d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<3x2xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_2d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<4x5x!FHE.eint<6>>, tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_2d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<2x3x2xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_3d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<4x5x!FHE.eint<6>>, tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_3d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<6x2x3x2xindex>) -> tensor<6x2x3x!FHE.eint<6>>
  return %output : tensor<6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_to_4d(%[[input:.*]]: tensor<4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<4x5x!FHE.eint<6>>, tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_to_4d(%input: tensor<4x5x!FHE.eint<6>>, %indices: tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<4x5x!FHE.eint<6>>, tensor<5x6x2x3x2xindex>) -> tensor<5x6x2x3x!FHE.eint<6>>
  return %output : tensor<5x6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_1d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<2x4x5x!FHE.eint<6>>, tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_1d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<3x3xindex>) -> tensor<3x!FHE.eint<6>>
  return %output : tensor<3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_2d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<2x4x5x!FHE.eint<6>>, tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_2d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<2x3x3xindex>) -> tensor<2x3x!FHE.eint<6>>
  return %output : tensor<2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_3d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<2x4x5x!FHE.eint<6>>, tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_3d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<6x2x3x3xindex>) -> tensor<6x2x3x!FHE.eint<6>>
  return %output : tensor<6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_4d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<4x6x2x3x3xindex>) -> tensor<4x6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<2x4x5x!FHE.eint<6>>, tensor<4x6x2x3x3xindex>) -> tensor<4x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<4x6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_4d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<4x6x2x3x3xindex>) -> tensor<4x6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<4x6x2x3x3xindex>) -> tensor<4x6x2x3x!FHE.eint<6>>
  return %output : tensor<4x6x2x3x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_to_5d(%[[input:.*]]: tensor<2x4x5x!FHE.eint<6>>, %[[indices:.*]]: tensor<5x4x6x2x3x3xindex>) -> tensor<5x4x6x2x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_index"(%[[input]], %[[indices]]) : (tensor<2x4x5x!FHE.eint<6>>, tensor<5x4x6x2x3x3xindex>) -> tensor<5x4x6x2x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x4x6x2x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_to_5d(%input: tensor<2x4x5x!FHE.eint<6>>, %indices: tensor<5x4x6x2x3x3xindex>) -> tensor<5x4x6x2x3x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_index"(%input, %indices) : (tensor<2x4x5x!FHE.eint<6>>, tensor<5x4x6x2x3x3xindex>) -> tensor<5x4x6x2x3x!FHE.eint<6>>
  return %output : tensor<5x4x6x2x3x!FHE.eint<6>>
}

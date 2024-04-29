// RUN: concretecompiler --split-input-file --action=roundtrip %s 2>&1| FileCheck %s

// -----

// CHECK:      func.func @from_1d_into_1d(%[[input:.*]]: tensor<25x!FHE.eint<6>>, %[[indices:.*]]: tensor<3xindex>, %[[values:.*]]: tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<25x!FHE.eint<6>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<25x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_into_1d(%input: tensor<25x!FHE.eint<6>>, %indices: tensor<3xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<25x!FHE.eint<6>>, tensor<3xindex>, tensor<3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
  return %output : tensor<25x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_into_1d(%[[input:.*]]: tensor<25x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3xindex>, %[[values:.*]]: tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<25x!FHE.eint<6>>, tensor<2x3xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<25x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_into_1d(%input: tensor<25x!FHE.eint<6>>, %indices: tensor<2x3xindex>, %values: tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<25x!FHE.eint<6>>, tensor<2x3xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
  return %output : tensor<25x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_3d_into_1d(%[[input:.*]]: tensor<25x!FHE.eint<6>>, %[[indices:.*]]: tensor<4x2x3xindex>, %[[values:.*]]:  tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<25x!FHE.eint<6>>, tensor<4x2x3xindex>, tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<25x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_into_1d(%input: tensor<25x!FHE.eint<6>>, %indices: tensor<4x2x3xindex>, %values: tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<25x!FHE.eint<6>>, tensor<4x2x3xindex>, tensor<4x2x3x!FHE.eint<6>>) -> tensor<25x!FHE.eint<6>>
  return %output : tensor<25x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_1d_into_2d(%[[input:.*]]: tensor<5x10x!FHE.eint<6>>, %[[indices:.*]]: tensor<3x2xindex>, %[[values:.*]]: tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<5x10x!FHE.eint<6>>, tensor<3x2xindex>, tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_1d_into_2d(%input: tensor<5x10x!FHE.eint<6>>, %indices: tensor<3x2xindex>, %values: tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x10x!FHE.eint<6>>, tensor<3x2xindex>, tensor<3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
  return %output : tensor<5x10x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_2d_into_2d(%[[input:.*]]: tensor<5x10x!FHE.eint<6>>, %[[indices:.*]]: tensor<2x3x2xindex>, %[[values:.*]]: tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<5x10x!FHE.eint<6>>, tensor<2x3x2xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_2d_into_2d(%input: tensor<5x10x!FHE.eint<6>>, %indices: tensor<2x3x2xindex>, %values: tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x10x!FHE.eint<6>>, tensor<2x3x2xindex>, tensor<2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
  return %output : tensor<5x10x!FHE.eint<6>>
}


// -----

// CHECK:      func.func @from_3d_into_2d(%[[input:.*]]: tensor<5x10x!FHE.eint<6>>, %[[indices:.*]]: tensor<6x2x3x2xindex>, %[[values:.*]]: tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<5x10x!FHE.eint<6>>, tensor<6x2x3x2xindex>, tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<5x10x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_3d_into_2d(%input: tensor<5x10x!FHE.eint<6>>, %indices: tensor<6x2x3x2xindex>, %values: tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<5x10x!FHE.eint<6>>, tensor<6x2x3x2xindex>, tensor<6x2x3x!FHE.eint<6>>) -> tensor<5x10x!FHE.eint<6>>
  return %output : tensor<5x10x!FHE.eint<6>>
}

// -----

// CHECK:      func.func @from_4d_into_3d(%[[input:.*]]: tensor<20x5x2x!FHE.eint<6>>, %[[indices:.*]]: tensor<5x6x2x3x3xindex>, %[[values:.*]]: tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>> {
// CHECK-NEXT:   %[[output:.*]] = "FHELinalg.fancy_assign"(%[[input]], %[[indices]], %[[values]]) : (tensor<20x5x2x!FHE.eint<6>>, tensor<5x6x2x3x3xindex>, tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>>
// CHECK-NEXT:   return %[[output]] : tensor<20x5x2x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @from_4d_into_3d(%input: tensor<20x5x2x!FHE.eint<6>>, %indices: tensor<5x6x2x3x3xindex>, %values: tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>> {
  %output = "FHELinalg.fancy_assign"(%input, %indices, %values) : (tensor<20x5x2x!FHE.eint<6>>, tensor<5x6x2x3x3xindex>, tensor<5x6x2x3x!FHE.eint<6>>) -> tensor<20x5x2x!FHE.eint<6>>
  return %output : tensor<20x5x2x!FHE.eint<6>>
}

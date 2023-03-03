// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1, d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel"]} ins(%[[a0]] : tensor<2x3x!FHE.eint<7>>) outs(%[[v0]] : tensor<3x2x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x3x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
  %0 = "FHELinalg.transpose"(%arg0) : (tensor<2x3x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>
  return %0 : tensor<3x2x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1, d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x2x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x3x2x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<2x3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<4x3x2x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<4x3x2x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x3x2x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x2x!FHE.eint<7>> {
  %0 = "FHELinalg.transpose"(%arg0) : (tensor<2x3x4x!FHE.eint<7>>) -> tensor<4x3x2x!FHE.eint<7>>
  return %0 : tensor<4x3x2x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d0, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x3x5x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<3x4x5x!FHE.eint<6>>) outs(%[[v0]] : tensor<4x3x5x!FHE.eint<6>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<6>, %[[aa1:.*]]: !FHE.eint<6>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<4x3x5x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x3x5x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>> {
  %0 = "FHELinalg.transpose"(%arg0) { axes = [1, 0, 2] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x3x5x!FHE.eint<6>>
  return %0 : tensor<4x3x5x!FHE.eint<6>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1, d2, d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x5x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x5x3x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<3x4x5x!FHE.eint<6>>) outs(%[[v0]] : tensor<4x5x3x!FHE.eint<6>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<6>, %[[aa1:.*]]: !FHE.eint<6>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<4x5x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x5x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x5x3x!FHE.eint<6>> {
  %0 = "FHELinalg.transpose"(%arg0) { axes = [1, 2, 0] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<4x5x3x!FHE.eint<6>>
  return %0 : tensor<4x5x3x!FHE.eint<6>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x5x!FHE.eint<6>>) -> tensor<3x5x4x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x5x4x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<3x4x5x!FHE.eint<6>>) outs(%[[v0]] : tensor<3x5x4x!FHE.eint<6>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<6>, %[[aa1:.*]]: !FHE.eint<6>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<3x5x4x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x5x4x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<3x5x4x!FHE.eint<6>> {
  %0 = "FHELinalg.transpose"(%arg0) { axes = [0, 2, 1] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<3x5x4x!FHE.eint<6>>
  return %0 : tensor<3x5x4x!FHE.eint<6>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d2, d0, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x3x4x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x4x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<3x4x5x!FHE.eint<6>>) outs(%[[v0]] : tensor<5x3x4x!FHE.eint<6>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<6>, %[[aa1:.*]]: !FHE.eint<6>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<5x3x4x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x4x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x3x4x!FHE.eint<6>> {
  %0 = "FHELinalg.transpose"(%arg0) { axes = [2, 0, 1] } : (tensor<3x4x5x!FHE.eint<6>>) -> tensor<5x3x4x!FHE.eint<6>>
  return %0 : tensor<5x3x4x!FHE.eint<6>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d0, d2, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x5x!FHE.eint<6>>) -> tensor<5x2x4x3x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x2x4x3x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<2x3x4x5x!FHE.eint<6>>) outs(%[[v0]] : tensor<5x2x4x3x!FHE.eint<6>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<6>, %[[aa1:.*]]: !FHE.eint<6>):
// CHECK-NEXT:       linalg.yield %[[aa0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<5x2x4x3x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x2x4x3x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x3x4x5x!FHE.eint<6>>) -> tensor<5x2x4x3x!FHE.eint<6>> {
  %0 = "FHELinalg.transpose"(%arg0) { axes = [3, 0, 2, 1] } : (tensor<2x3x4x5x!FHE.eint<6>>) -> tensor<5x2x4x3x!FHE.eint<6>>
  return %0 : tensor<5x2x4x3x!FHE.eint<6>>
}

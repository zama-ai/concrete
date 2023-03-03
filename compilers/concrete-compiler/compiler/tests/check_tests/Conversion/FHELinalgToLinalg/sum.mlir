// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero"() : () -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<0x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<0x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero"() : () -> !FHE.eint<7>
// CHECK-NEXT:   return %[[v0]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<3x0x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x4x!FHE.eint<7>>
  return %0 : tensor<3x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x1x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x1x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x1x4x!FHE.eint<7>>
  return %0 : tensor<3x1x4x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x0x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x0x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [2] } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x!FHE.eint<7>>
  return %0 : tensor<3x0x!FHE.eint<7>>
}

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x0x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v0]] : tensor<3x0x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [2], keep_dims = true } : (tensor<3x0x4x!FHE.eint<7>>) -> tensor<3x0x1x!FHE.eint<7>>
  return %0 : tensor<3x0x1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction"]} ins(%[[a0]] : tensor<4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v2:.*]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction"]} ins(%[[a0]] : tensor<4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v2:.*]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0] } : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction"]} ins(%[[a0]] : tensor<4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction"]} ins(%[[a0]] : tensor<4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0], keep_dims = true } : (tensor<4x!FHE.eint<7>>) -> tensor<1x!FHE.eint<7>>
  return %0 : tensor<1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v2:.*]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, 0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<4x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0] } : (tensor<3x4x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x4x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x4x!FHE.eint<7>>
  return %0 : tensor<1x4x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<3x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<3x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x!FHE.eint<7>>
  return %0 : tensor<3x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (d0, 0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<3x1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<3x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<3x1x!FHE.eint<7>>
  return %0 : tensor<3x1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v2:.*]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1] } : (tensor<3x4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1) -> (0, 0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1], keep_dims = true } : (tensor<3x4x!FHE.eint<7>>) -> tensor<1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v2:.*]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (0, 0, 0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x1x1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<3x2x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x2x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x2x!FHE.eint<7>>
  return %0 : tensor<3x2x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d0, 0, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x1x2x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<3x1x2x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<3x1x2x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x1x2x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [1], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<3x1x2x!FHE.eint<7>>
  return %0 : tensor<3x1x2x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel", "reduction"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<4x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<4x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<4x!FHE.eint<7>>
  return %0 : tensor<4x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (0, d1, 0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x4x1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "parallel", "reduction"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x4x1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x4x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x4x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x4x1x!FHE.eint<7>>
  return %0 : tensor<1x4x1x!FHE.eint<7>>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v2:.*]] = tensor.extract %[[v1]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v2]] : !FHE.eint<7>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2] } : (tensor<3x4x2x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

// -----

// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2) -> (0, 0, 0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction", "reduction", "reduction"]} ins(%[[a0]] : tensor<3x4x2x!FHE.eint<7>>) outs(%[[v0]] : tensor<1x1x1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v1]] : tensor<1x1x1x!FHE.eint<7>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>> {
  %0 = "FHELinalg.sum"(%arg0) { axes = [0, 1, 2], keep_dims = true } : (tensor<3x4x2x!FHE.eint<7>>) -> tensor<1x1x1x!FHE.eint<7>>
  return %0 : tensor<1x1x1x!FHE.eint<7>>
}

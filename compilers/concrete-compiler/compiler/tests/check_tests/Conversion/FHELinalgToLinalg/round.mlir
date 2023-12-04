// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK: #[[m0:.*]] = affine_map<(d0) -> (d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m0]]], iterator_types = ["parallel"]} ins(%[[a0]] : tensor<5x!FHE.eint<8>>) outs(%[[v0]] : tensor<5x!FHE.eint<6>>) {
// CHECK-NEXT:   ^bb0(%[[i0:.*]]: !FHE.eint<8>, %[[o0:.*]]: !FHE.eint<6>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.round"(%[[i0]]) : (!FHE.eint<8>) -> !FHE.eint<6>
// CHECK-NEXT:     linalg.yield %[[vv0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<5x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<5x!FHE.eint<8>>) -> tensor<5x!FHE.eint<6>> {
  %1 = "FHELinalg.round"(%arg0): (tensor<5x!FHE.eint<8>>) -> (tensor<5x!FHE.eint<6>>)
  return %1: tensor<5x!FHE.eint<6>>
}

// -----

// CHECK: #[[m0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.eint<8>>) -> tensor<2x3x4x!FHE.eint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x3x4x!FHE.eint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<2x3x4x!FHE.eint<8>>) outs(%[[v0]] : tensor<2x3x4x!FHE.eint<6>>) {
// CHECK-NEXT:   ^bb0(%[[i0:.*]]: !FHE.eint<8>, %[[o0:.*]]: !FHE.eint<6>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.round"(%[[i0]]) : (!FHE.eint<8>) -> !FHE.eint<6>
// CHECK-NEXT:     linalg.yield %[[vv0]] : !FHE.eint<6>
// CHECK-NEXT:   } -> tensor<2x3x4x!FHE.eint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<2x3x4x!FHE.eint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x3x4x!FHE.eint<8>>) -> tensor<2x3x4x!FHE.eint<6>> {
  %1 = "FHELinalg.round"(%arg0): (tensor<2x3x4x!FHE.eint<8>>) -> (tensor<2x3x4x!FHE.eint<6>>)
  return %1: tensor<2x3x4x!FHE.eint<6>>
}

// -----

// CHECK: #[[m0:.*]] = affine_map<(d0) -> (d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x!FHE.esint<8>>) -> tensor<5x!FHE.esint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x!FHE.esint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m0]]], iterator_types = ["parallel"]} ins(%[[a0]] : tensor<5x!FHE.esint<8>>) outs(%[[v0]] : tensor<5x!FHE.esint<6>>) {
// CHECK-NEXT:   ^bb0(%[[i0:.*]]: !FHE.esint<8>, %[[o0:.*]]: !FHE.esint<6>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.round"(%[[i0]]) : (!FHE.esint<8>) -> !FHE.esint<6>
// CHECK-NEXT:     linalg.yield %[[vv0]] : !FHE.esint<6>
// CHECK-NEXT:   } -> tensor<5x!FHE.esint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x!FHE.esint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<5x!FHE.esint<8>>) -> tensor<5x!FHE.esint<6>> {
  %1 = "FHELinalg.round"(%arg0): (tensor<5x!FHE.esint<8>>) -> (tensor<5x!FHE.esint<6>>)
  return %1: tensor<5x!FHE.esint<6>>
}

// -----

// CHECK: #[[m0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x3x4x!FHE.esint<8>>) -> tensor<2x3x4x!FHE.esint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x3x4x!FHE.esint<6>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m0]]], iterator_types = ["parallel", "parallel", "parallel"]} ins(%[[a0]] : tensor<2x3x4x!FHE.esint<8>>) outs(%[[v0]] : tensor<2x3x4x!FHE.esint<6>>) {
// CHECK-NEXT:   ^bb0(%[[i0:.*]]: !FHE.esint<8>, %[[o0:.*]]: !FHE.esint<6>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.round"(%[[i0]]) : (!FHE.esint<8>) -> !FHE.esint<6>
// CHECK-NEXT:     linalg.yield %[[vv0]] : !FHE.esint<6>
// CHECK-NEXT:   } -> tensor<2x3x4x!FHE.esint<6>>
// CHECK-NEXT:   return %[[v1]] : tensor<2x3x4x!FHE.esint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<2x3x4x!FHE.esint<8>>) -> tensor<2x3x4x!FHE.esint<6>> {
  %1 = "FHELinalg.round"(%arg0): (tensor<2x3x4x!FHE.esint<8>>) -> (tensor<2x3x4x!FHE.esint<6>>)
  return %1: tensor<2x3x4x!FHE.esint<6>>
}

// RUN: concretecompiler %s --action=dump-tfhe --passes fhe-tensor-ops-to-linalg 2>&1 | FileCheck %s

// CHECK: #[[$MAP0:.*]] = affine_map<(d0) -> (d0)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0) -> (0)>

// CHECK:      func @sum_1D(%[[a0:.*]]: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero"() : () -> !FHE.eint<7>
// CHECK-NEXT:   %[[v1:.*]] = tensor.from_elements %[[v0]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[v2:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]]], iterator_types = ["reduction"]} ins(%[[a0]] : tensor<4x!FHE.eint<7>>) outs(%[[v1]] : tensor<1x!FHE.eint<7>>) {
// CHECK-NEXT:     ^bb0(%[[aa0:.*]]: !FHE.eint<7>, %[[aa1:.*]]: !FHE.eint<7>):
// CHECK-NEXT:       %[[vv0:.*]] = "FHE.add_eint"(%[[aa0]], %[[aa1]]) : (!FHE.eint<7>, !FHE.eint<7>) -> !FHE.eint<7>
// CHECK-NEXT:       linalg.yield %[[vv0]] : !FHE.eint<7>
// CHECK-NEXT:   } -> tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   %[[c0:.*]] = arith.constant 0 : index
// CHECK-NEXT:   %[[v3:.*]] = tensor.extract %[[v2]][%[[c0]]] : tensor<1x!FHE.eint<7>>
// CHECK-NEXT:   return %[[v3]] : !FHE.eint<7>
// CHECK-NEXT: }
func @sum_1D(%arg0: tensor<4x!FHE.eint<7>>) -> !FHE.eint<7> {
  %0 = "FHELinalg.sum"(%arg0) : (tensor<4x!FHE.eint<7>>) -> !FHE.eint<7>
  return %0 : !FHE.eint<7>
}

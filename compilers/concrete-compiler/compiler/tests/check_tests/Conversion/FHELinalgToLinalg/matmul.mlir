// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<4x2xi6>) -> tensor<3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<3x4x!FHE.eint<5>>, tensor<4x2xi6>) outs(%[[v0]] : tensor<3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<5>>, %y: tensor<4x2xi6>) -> tensor<3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x4x!FHE.eint<5>>, tensor<4x2xi6>) -> tensor<3x2x!FHE.eint<5>>
  return %0 : tensor<3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1) -> (d0)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1) -> (d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<5>>, %[[a1:.*]]: tensor<3x2xi6>) -> tensor<2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["reduction", "parallel"]} ins(%[[a0]], %[[a1]] : tensor<3x!FHE.eint<5>>, tensor<3x2xi6>) outs(%[[v0]] : tensor<2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<5>>, %y: tensor<3x2xi6>) -> tensor<2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<5>>, tensor<3x2xi6>) -> tensor<2x!FHE.eint<5>>
  return %0 : tensor<2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2) -> (d1)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<5>>, %[[a1:.*]]: tensor<5x3x2xi6>) -> tensor<5x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "reduction", "parallel"]} ins(%[[a0]], %[[a1]] : tensor<3x!FHE.eint<5>>, tensor<5x3x2xi6>) outs(%[[v0]] : tensor<5x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<5>>, %y: tensor<5x3x2xi6>) -> tensor<5x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<5>>, tensor<5x3x2xi6>) -> tensor<5x2x!FHE.eint<5>>
  return %0 : tensor<5x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x!FHE.eint<5>>, %[[a1:.*]]: tensor<4x5x3x2xi6>) -> tensor<4x5x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x5x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "reduction", "parallel"]} ins(%[[a0]], %[[a1]] : tensor<3x!FHE.eint<5>>, tensor<4x5x3x2xi6>) outs(%[[v0]] : tensor<4x5x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<4x5x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x5x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x!FHE.eint<5>>, %y: tensor<4x5x3x2xi6>) -> tensor<4x5x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x!FHE.eint<5>>, tensor<4x5x3x2xi6>) -> tensor<4x5x2x!FHE.eint<5>>
  return %0 : tensor<4x5x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1) -> (d1)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1) -> (d0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x2x!FHE.eint<5>>, %[[a1:.*]]: tensor<2xi6>) -> tensor<3x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<3x2x!FHE.eint<5>>, tensor<2xi6>) outs(%[[v0]] : tensor<3x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<3x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x2x!FHE.eint<5>>, %y: tensor<2xi6>) -> tensor<3x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x2x!FHE.eint<5>>, tensor<2xi6>) -> tensor<3x!FHE.eint<5>>
  return %0 : tensor<3x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x3x2x!FHE.eint<5>>, %[[a1:.*]]: tensor<2xi6>) -> tensor<5x3x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<5x3x2x!FHE.eint<5>>, tensor<2xi6>) outs(%[[v0]] : tensor<5x3x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x3x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x3x2x!FHE.eint<5>>, %y: tensor<2xi6>) -> tensor<5x3x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x3x2x!FHE.eint<5>>, tensor<2xi6>) -> tensor<5x3x!FHE.eint<5>>
  return %0 : tensor<5x3x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<4x5x3x2x!FHE.eint<5>>, %[[a1:.*]]: tensor<2xi6>) -> tensor<4x5x3x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x5x3x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<4x5x3x2x!FHE.eint<5>>, tensor<2xi6>) outs(%[[v0]] : tensor<4x5x3x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<4x5x3x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<4x5x3x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<4x5x3x2x!FHE.eint<5>>, %y: tensor<2xi6>) -> tensor<4x5x3x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<4x5x3x2x!FHE.eint<5>>, tensor<2xi6>) -> tensor<4x5x3x!FHE.eint<5>>
  return %0 : tensor<4x5x3x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<5x3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) outs(%[[v0]] : tensor<5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x3x4x!FHE.eint<5>>, %y: tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>>
  return %0 : tensor<5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d3, d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<1x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<5x3x4x!FHE.eint<5>>, tensor<1x4x2xi6>) outs(%[[v0]] : tensor<5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x3x4x!FHE.eint<5>>, %y: tensor<1x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x3x4x!FHE.eint<5>>, tensor<1x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>>
  return %0 : tensor<5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (0, d1, d3)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<1x3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<1x3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) outs(%[[v0]] : tensor<5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<1x3x4x!FHE.eint<5>>, %y: tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<1x3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>>
  return %0 : tensor<5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (d3, d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<5x3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<5x3x4x!FHE.eint<5>>, tensor<4x2xi6>) outs(%[[v0]] : tensor<5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<5x3x4x!FHE.eint<5>>, %y: tensor<4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<5x3x4x!FHE.eint<5>>, tensor<4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>>
  return %0 : tensor<5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3) -> (d1, d3)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) outs(%[[v0]] : tensor<5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<5>>, %y: tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) -> tensor<5x3x2x!FHE.eint<5>>
  return %0 : tensor<5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d4, d3)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x5x3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<2x5x4x2xi6>) -> tensor<2x5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<2x5x3x4x!FHE.eint<5>>, tensor<2x5x4x2xi6>) outs(%[[v0]] : tensor<2x5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<2x5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<2x5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x5x3x4x!FHE.eint<5>>, %y: tensor<2x5x4x2xi6>) -> tensor<2x5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x5x3x4x!FHE.eint<5>>, tensor<2x5x4x2xi6>) -> tensor<2x5x3x2x!FHE.eint<5>>
  return %0 : tensor<2x5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, 0, d2, d4)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d1, d4, d3)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x1x3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<5x4x2xi6>) -> tensor<2x5x3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<2x1x3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) outs(%[[v0]] : tensor<2x5x3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<2x5x3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<2x5x3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x1x3x4x!FHE.eint<5>>, %y: tensor<5x4x2xi6>) -> tensor<2x5x3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x1x3x4x!FHE.eint<5>>, tensor<5x4x2xi6>) -> tensor<2x5x3x2x!FHE.eint<5>>
  return %0 : tensor<2x5x3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d4)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (0, d4, d3)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<2x5x4x3x!FHE.eint<5>>, %[[a1:.*]]: tensor<1x3x2xi6>) -> tensor<2x5x4x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<2x5x4x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<2x5x4x3x!FHE.eint<5>>, tensor<1x3x2xi6>) outs(%[[v0]] : tensor<2x5x4x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: i6, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<2x5x4x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<2x5x4x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<2x5x4x3x!FHE.eint<5>>, %y: tensor<1x3x2xi6>) -> tensor<2x5x4x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_int"(%x, %y): (tensor<2x5x4x3x!FHE.eint<5>>, tensor<1x3x2xi6>) -> tensor<2x5x4x2x!FHE.eint<5>>
  return %0 : tensor<2x5x4x2x!FHE.eint<5>>
}

// -----

// CHECK:      #[[m0:.*]] = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT: #[[m1:.*]] = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT: #[[m2:.*]] = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4xi6>, %[[a1:.*]]: tensor<4x2x!FHE.eint<5>>) -> tensor<3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<3x4xi6>, tensor<4x2x!FHE.eint<5>>) outs(%[[v0]] : tensor<3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: i6, %[[aa1:.*]]: !FHE.eint<5>, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint_int"(%[[aa1]], %[[aa0]]) : (!FHE.eint<5>, i6) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4xi6>, %y: tensor<4x2x!FHE.eint<5>>) -> tensor<3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_int_eint"(%x, %y): (tensor<3x4xi6>, tensor<4x2x!FHE.eint<5>>) -> tensor<3x2x!FHE.eint<5>>
  return %0 : tensor<3x2x!FHE.eint<5>>
}

// -----

// CHECK:      #map = affine_map<(d0, d1, d2) -> (d0, d2)>
// CHECK-NEXT: #map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
// CHECK-NEXT: #map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<3x4x!FHE.eint<5>>, %[[a1:.*]]: tensor<4x2x!FHE.eint<5>>) -> tensor<3x2x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = linalg.generic {indexing_maps = [#[[m0]], #[[m1]], #[[m2]]], iterator_types = ["parallel", "parallel", "reduction"]} ins(%[[a0]], %[[a1]] : tensor<3x4x!FHE.eint<5>>, tensor<4x2x!FHE.eint<5>>) outs(%[[v0]] : tensor<3x2x!FHE.eint<5>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.eint<5>, %[[aa1:.*]]: !FHE.eint<5>, %[[aa2:.*]]: !FHE.eint<5>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.mul_eint"(%[[aa1]], %[[aa0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     %[[vv1:.*]] = "FHE.add_eint"(%[[aa2]], %[[vv0]]) : (!FHE.eint<5>, !FHE.eint<5>) -> !FHE.eint<5>
// CHECK-NEXT:     linalg.yield %[[vv1]] : !FHE.eint<5>
// CHECK-NEXT:   } -> tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v1]] : tensor<3x2x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%x: tensor<3x4x!FHE.eint<5>>, %y: tensor<4x2x!FHE.eint<5>>) -> tensor<3x2x!FHE.eint<5>> {
  %0 = "FHELinalg.matmul_eint_eint"(%x, %y): (tensor<3x4x!FHE.eint<5>>, tensor<4x2x!FHE.eint<5>>) -> tensor<3x2x!FHE.eint<5>>
  return %0 : tensor<3x2x!FHE.eint<5>>
}

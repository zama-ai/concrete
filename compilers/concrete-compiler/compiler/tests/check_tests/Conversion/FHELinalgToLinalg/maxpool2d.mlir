// RUN: concretecompiler --split-input-file --action=dump-tfhe --passes fhe-tensor-ops-to-linalg %s 2>&1 | FileCheck %s

// -----

// CHECK:      func.func @main(%[[a0:.*]]: tensor<1x1x8x10x!FHE.eint<5>>) -> tensor<1x1x6x9x!FHE.eint<5>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x6x9x!FHE.eint<5>>
// CHECK-NEXT:   %[[v1:.*]] = tensor.empty() : tensor<3x2xi64>
// CHECK-NEXT:   %[[v2:.*]] = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, max_signed = {op = "FHE.max_eint", op_attrs = {}}, strides = dense<1> : vector<2xi64>} ins(%arg0, %1 : tensor<1x1x8x10x!FHE.eint<5>>, tensor<3x2xi64>) outs(%0 : tensor<1x1x6x9x!FHE.eint<5>>) -> tensor<1x1x6x9x!FHE.eint<5>>
// CHECK-NEXT:   return %[[v2]] : tensor<1x1x6x9x!FHE.eint<5>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<1x1x8x10x!FHE.eint<5>>) -> tensor<1x1x6x9x!FHE.eint<5>> {
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[3, 2]> : tensor<2xi64> } : (tensor<1x1x8x10x!FHE.eint<5>>) -> tensor<1x1x6x9x!FHE.eint<5>>
  return %0 : tensor<1x1x6x9x!FHE.eint<5>>
}

// -----


// CHECK: #[[$MAP0:.*]] = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
// CHECK: #[[$MAP1:.*]] = affine_map<(d0, d1, d2, d3) -> (0)>

// CHECK:      func.func @main(%[[a0:.*]]: tensor<1x1x6x5x!FHE.esint<6>>) -> tensor<1x1x5x3x!FHE.esint<6>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x5x3x!FHE.esint<6>>
// CHECK-NEXT:   %[[v1:.*]] = arith.constant dense<16> : tensor<1xi7>
// CHECK-NEXT:   %[[v2:.*]] = "FHE.zero_tensor"() : () -> tensor<1x1x5x3x!FHE.esint<6>>
// CHECK-NEXT:   %[[v3:.*]] = linalg.generic {indexing_maps = [#[[$MAP0]], #[[$MAP1]], #[[$MAP0]]], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%[[v0]], %[[v1]] : tensor<1x1x5x3x!FHE.esint<6>>, tensor<1xi7>) outs(%[[v2]] : tensor<1x1x5x3x!FHE.esint<6>>) {
// CHECK-NEXT:   ^bb0(%[[aa0:.*]]: !FHE.esint<6>, %[[aa1:.*]]: i7, %[[aa2:.*]]: !FHE.esint<6>):
// CHECK-NEXT:     %[[vv0:.*]] = "FHE.sub_eint_int"(%[[aa0]], %[[aa1]]) : (!FHE.esint<6>, i7) -> !FHE.esint<6>
// CHECK-NEXT:     linalg.yield %[[vv0]] : !FHE.esint<6>
// CHECK-NEXT:   } -> tensor<1x1x5x3x!FHE.esint<6>>
// CHECK-NEXT:   %[[v4:.*]] = tensor.empty() : tensor<2x3xi64>
// CHECK-NEXT:   %[[v5:.*]] = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, max_signed = {op = "FHE.max_eint", op_attrs = {}}, strides = dense<1> : vector<2xi64>} ins(%arg0, %[[v4]] : tensor<1x1x6x5x!FHE.esint<6>>, tensor<2x3xi64>) outs(%[[v3]] : tensor<1x1x5x3x!FHE.esint<6>>) -> tensor<1x1x5x3x!FHE.esint<6>>
// CHECK-NEXT:   return %[[v5]] : tensor<1x1x5x3x!FHE.esint<6>>
// CHECK-NEXT: }
func.func @main(%arg0: tensor<1x1x6x5x!FHE.esint<6>>) -> tensor<1x1x5x3x!FHE.esint<6>> {
  %0 = "FHELinalg.maxpool2d"(%arg0) { kernel_shape = dense<[2, 3]> : tensor<2xi64> } : (tensor<1x1x6x5x!FHE.esint<6>>) -> tensor<1x1x5x3x!FHE.esint<6>>
  return %0 : tensor<1x1x5x3x!FHE.esint<6>>
}

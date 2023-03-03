// RUN: concretecompiler --chunk-integers --chunk-size 4 --chunk-width 2 --passes fhe-big-int-transform --action=dump-fhe  %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_chunked_eint(%arg0: tensor<32x!FHE.eint<4>>, %arg1: tensor<32x!FHE.eint<4>>) -> tensor<32x!FHE.eint<4>>
func.func @add_chunked_eint(%arg0: !FHE.eint<64>, %arg1: !FHE.eint<64>) -> !FHE.eint<64> {
  // CHECK-NEXT: %[[V0:.*]] = "FHE.zero"() : () -> !FHE.eint<4>
  // CHECK-NEXT: %[[V1:.*]] = "FHE.zero_tensor"() : () -> tensor<32x!FHE.eint<4>>
  // CHECK-NEXT: %[[c4_i5:.*]] = arith.constant 4 : i5
  // CHECK-NEXT: %[[V2:.*]] = affine.for %arg2 = 0 to 32 iter_args(%arg3 = %[[V1]]) -> (tensor<32x!FHE.eint<4>>) {
  // CHECK-NEXT:   %[[V3:.*]] = tensor.extract %arg0[%arg2] : tensor<32x!FHE.eint<4>>
  // CHECK-NEXT:   %[[V4:.*]] = tensor.extract %arg1[%arg2] : tensor<32x!FHE.eint<4>>
  // CHECK-NEXT:   %[[V5:.*]] = "FHE.add_eint"(%[[V3]], %[[V4]]) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  // CHECK-NEXT:   %[[V6:.*]] = "FHE.add_eint"(%[[V5]], %[[V0]]) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  // CHECK-NEXT:   %[[cst:.*]] = arith.constant dense<[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]> : tensor<16xi64>
  // CHECK-NEXT:   %[[V7:.*]] = "FHE.apply_lookup_table"(%[[V6]], %[[cst]]) : (!FHE.eint<4>, tensor<16xi64>) -> !FHE.eint<4>
  // CHECK-NEXT:   %[[V8:.*]] = "FHE.mul_eint_int"(%[[V7]], %[[c4_i5]]) : (!FHE.eint<4>, i5) -> !FHE.eint<4>
  // CHECK-NEXT:   %[[V9:.*]] = "FHE.sub_eint"(%[[V6]], %[[V8]]) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  // CHECK-NEXT:   %[[V10:.*]] = tensor.insert %[[V9]] into %arg3[%arg2] : tensor<32x!FHE.eint<4>>
  // CHECK-NEXT:   affine.yield %[[V10]] : tensor<32x!FHE.eint<4>>
  // CHECK-NEXT: }
  // CHECK-NEXT: return %2 : tensor<32x!FHE.eint<4>>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<64>, !FHE.eint<64>) -> (!FHE.eint<64>)
  return %1: !FHE.eint<64>
}

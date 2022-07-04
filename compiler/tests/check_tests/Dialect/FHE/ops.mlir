// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK: func.func @zero() -> !FHE.eint<2>
func.func @zero() -> !FHE.eint<2> {
  // CHECK-NEXT: %[[RET:.*]] = "FHE.zero"() : () -> !FHE.eint<2>
  // CHECK-NEXT: return %[[RET]] : !FHE.eint<2>

  %1 = "FHE.zero"() : () -> !FHE.eint<2>
  return %1: !FHE.eint<2>
}

// CHECK: func.func @zero_1D() -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @zero_1D() -> tensor<4x!FHE.eint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<2>>
  return %0 : tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @zero_2D() -> tensor<4x9x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x9x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x9x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @zero_2D() -> tensor<4x9x!FHE.eint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x9x!FHE.eint<2>>
  return %0 : tensor<4x9x!FHE.eint<2>>
}

// CHECK-LABEL: func.func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%arg0, %[[V1]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.eint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @sub_int_eint(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @sub_int_eint(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.sub_int_eint"(%[[V1]], %arg0) : (i3, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.eint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i3, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @sub_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @sub_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.sub_eint_int"(%arg0, %[[V1]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.eint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.sub_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @sub_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2>
func.func @sub_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.sub_eint"(%arg0, %arg1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @neg_eint(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @neg_eint(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.neg_eint"(%arg0) : (!FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.neg_eint"(%arg0): (!FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @mul_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @mul_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%arg0, %[[V1]]) : (!FHE.eint<2>, i3) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.eint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @add_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2>
func.func @add_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint"(%arg0, %arg1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2>
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.apply_lookup_table"(%arg0, %arg1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

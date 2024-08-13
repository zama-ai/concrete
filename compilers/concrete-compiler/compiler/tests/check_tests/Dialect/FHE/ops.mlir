// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK: func.func @zero() -> !FHE.eint<2>
func.func @zero() -> !FHE.eint<2> {
  // CHECK-NEXT: %[[RET:.*]] = "FHE.zero"() : () -> !FHE.eint<2>
  // CHECK-NEXT: return %[[RET]] : !FHE.eint<2>

  %1 = "FHE.zero"() : () -> !FHE.eint<2>
  return %1: !FHE.eint<2>
}

// CHECK: func.func @zero_signed() -> !FHE.esint<2>
func.func @zero_signed() -> !FHE.esint<2> {
  // CHECK-NEXT: %[[RET:.*]] = "FHE.zero"() : () -> !FHE.esint<2>
  // CHECK-NEXT: return %[[RET]] : !FHE.esint<2>

  %1 = "FHE.zero"() : () -> !FHE.esint<2>
  return %1: !FHE.esint<2>
}

// CHECK: func.func @zero_1D() -> tensor<4x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @zero_1D() -> tensor<4x!FHE.eint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x!FHE.eint<2>>
  return %0 : tensor<4x!FHE.eint<2>>
}

// CHECK: func.func @zero_1D_signed() -> tensor<4x!FHE.esint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x!FHE.esint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x!FHE.esint<2>>
// CHECK-NEXT: }
func.func @zero_1D_signed() -> tensor<4x!FHE.esint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x!FHE.esint<2>>
  return %0 : tensor<4x!FHE.esint<2>>
}

// CHECK: func.func @zero_2D() -> tensor<4x9x!FHE.eint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x9x!FHE.eint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x9x!FHE.eint<2>>
// CHECK-NEXT: }
func.func @zero_2D() -> tensor<4x9x!FHE.eint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x9x!FHE.eint<2>>
  return %0 : tensor<4x9x!FHE.eint<2>>
}

// CHECK: func.func @zero_2D_signed() -> tensor<4x9x!FHE.esint<2>> {
// CHECK-NEXT:   %[[v0:.*]] = "FHE.zero_tensor"() : () -> tensor<4x9x!FHE.esint<2>>
// CHECK-NEXT:   return %[[v0]] : tensor<4x9x!FHE.esint<2>>
// CHECK-NEXT: }
func.func @zero_2D_signed() -> tensor<4x9x!FHE.esint<2>> {
  %0 = "FHE.zero_tensor"() : () -> tensor<4x9x!FHE.esint<2>>
  return %0 : tensor<4x9x!FHE.esint<2>>
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

// CHECK-LABEL: func.func @add_eint_int_small_clear(%arg0: !FHE.eint<8>) -> !FHE.eint<8>
func.func @add_eint_int_small_clear(%arg0: !FHE.eint<8>) -> !FHE.eint<8> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i2
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%arg0, %[[V1]]) : (!FHE.eint<8>, i2) -> !FHE.eint<8>
  // CHECK-NEXT: return %[[V2]] : !FHE.eint<8>

  %0 = arith.constant 1 : i2
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<8>, i2) -> (!FHE.eint<8>)
  return %1: !FHE.eint<8>
}

// CHECK-LABEL: func.func @add_eint_int_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2>
func.func @add_eint_int_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%arg0, %[[V1]]) : (!FHE.esint<2>, i3) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.esint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.esint<2>, i3) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}

// CHECK-LABEL: func.func @add_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2>
func.func @add_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint"(%arg0, %arg1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @add_eint_signed(%arg0: !FHE.esint<2>, %arg1: !FHE.esint<2>) -> !FHE.esint<2>
func.func @add_eint_signed(%arg0: !FHE.esint<2>, %arg1: !FHE.esint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint"(%arg0, %arg1) : (!FHE.esint<2>, !FHE.esint<2>) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.esint<2>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.esint<2>, !FHE.esint<2>) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
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

// CHECK-LABEL: func.func @sub_int_eint_small_clear(%arg0: !FHE.esint<4>) -> !FHE.esint<4>
func.func @sub_int_eint_small_clear(%arg0: !FHE.esint<4>) -> !FHE.esint<4> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 3 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.sub_int_eint"(%[[V1]], %arg0) : (i3, !FHE.esint<4>) -> !FHE.esint<4>
  // CHECK-NEXT: return %[[V2]] : !FHE.esint<4>

  %0 = arith.constant 3 : i3
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i3, !FHE.esint<4>) -> (!FHE.esint<4>)
  return %1: !FHE.esint<4>
}

// CHECK-LABEL: func.func @sub_int_eint_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2>
func.func @sub_int_eint_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.sub_int_eint"(%[[V1]], %arg0) : (i3, !FHE.esint<2>) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.esint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.sub_int_eint"(%0, %arg0): (i3, !FHE.esint<2>) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
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

// CHECK-LABEL: func.func @sub_eint_int_small_clear(%arg0: !FHE.esint<4>) -> !FHE.esint<4>
func.func @sub_eint_int_small_clear(%arg0: !FHE.esint<4>) -> !FHE.esint<4> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 3 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.sub_eint_int"(%arg0, %[[V1]]) : (!FHE.esint<4>, i3) -> !FHE.esint<4>
  // CHECK-NEXT: return %[[V2]] : !FHE.esint<4>

  %0 = arith.constant 3 : i3
  %1 = "FHE.sub_eint_int"(%arg0, %0): (!FHE.esint<4>, i3) -> (!FHE.esint<4>)
  return %1: !FHE.esint<4>
}

// CHECK-LABEL: func.func @sub_eint_int_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2>
func.func @sub_eint_int_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.sub_eint_int"(%arg0, %[[V1]]) : (!FHE.esint<2>, i3) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.esint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.sub_eint_int"(%arg0, %0): (!FHE.esint<2>, i3) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}

// CHECK-LABEL: func.func @sub_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2>
func.func @sub_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.sub_eint"(%arg0, %arg1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @sub_eint_signed(%arg0: !FHE.esint<2>, %arg1: !FHE.esint<2>) -> !FHE.esint<2>
func.func @sub_eint_signed(%arg0: !FHE.esint<2>, %arg1: !FHE.esint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.sub_eint"(%arg0, %arg1) : (!FHE.esint<2>, !FHE.esint<2>) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.esint<2>

  %1 = "FHE.sub_eint"(%arg0, %arg1): (!FHE.esint<2>, !FHE.esint<2>) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
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

// CHECK-LABEL: func.func @mul_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2>
func.func @mul_eint(%arg0: !FHE.eint<2>, %arg1: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V0:.*]] = "FHE.mul_eint"(%arg0, %arg1) : (!FHE.eint<2>, !FHE.eint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V0]] : !FHE.eint<2>

  %0 = "FHE.mul_eint"(%arg0, %arg1): (!FHE.eint<2>, !FHE.eint<2>) -> (!FHE.eint<2>)
  return %0: !FHE.eint<2>
}

// CHECK-LABEL: func.func @mul_eint_int_small_clear(%arg0: !FHE.eint<4>) -> !FHE.eint<4>
func.func @mul_eint_int_small_clear(%arg0: !FHE.eint<4>) -> !FHE.eint<4> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 3 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%arg0, %[[V1]]) : (!FHE.eint<4>, i3) -> !FHE.eint<4>
  // CHECK-NEXT: return %[[V2]] : !FHE.eint<4>

  %0 = arith.constant 3 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<4>, i3) -> (!FHE.eint<4>)
  return %1: !FHE.eint<4>
}

// CHECK-LABEL: func.func @mul_eint_int_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2>
func.func @mul_eint_int_signed(%arg0: !FHE.esint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%arg0, %[[V1]]) : (!FHE.esint<2>, i3) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V2]] : !FHE.esint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.esint<2>, i3) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}

// CHECK-LABEL: func.func @to_signed(%arg0: !FHE.eint<2>) -> !FHE.esint<2>
func.func @to_signed(%arg0: !FHE.eint<2>) -> !FHE.esint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.to_signed"(%arg0) : (!FHE.eint<2>) -> !FHE.esint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.esint<2>

  %1 = "FHE.to_signed"(%arg0): (!FHE.eint<2>) -> (!FHE.esint<2>)
  return %1: !FHE.esint<2>
}

// CHECK-LABEL: func.func @to_unsigned(%arg0: !FHE.esint<2>) -> !FHE.eint<2>
func.func @to_unsigned(%arg0: !FHE.esint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.to_unsigned"(%arg0) : (!FHE.esint<2>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.to_unsigned"(%arg0): (!FHE.esint<2>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2>
func.func @apply_lookup_table(%arg0: !FHE.eint<2>, %arg1: tensor<4xi64>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.apply_lookup_table"(%arg0, %arg1) : (!FHE.eint<2>, tensor<4xi64>) -> !FHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<2>

  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<2>, tensor<4xi64>) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @max_eint(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4>
func.func @max_eint(%arg0: !FHE.eint<4>, %arg1: !FHE.eint<4>) -> !FHE.eint<4> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.max_eint"(%arg0, %arg1) : (!FHE.eint<4>, !FHE.eint<4>) -> !FHE.eint<4>
  %0 = "FHE.max_eint"(%arg0, %arg1): (!FHE.eint<4>, !FHE.eint<4>) -> (!FHE.eint<4>)

  // CHECK-NEXT: return %[[v0]] : !FHE.eint<4>
  return %0: !FHE.eint<4>
}

// CHECK-LABEL: func.func @max_esint(%arg0: !FHE.esint<4>, %arg1: !FHE.esint<4>) -> !FHE.esint<4>
func.func @max_esint(%arg0: !FHE.esint<4>, %arg1: !FHE.esint<4>) -> !FHE.esint<4> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.max_eint"(%arg0, %arg1) : (!FHE.esint<4>, !FHE.esint<4>) -> !FHE.esint<4>
  %0 = "FHE.max_eint"(%arg0, %arg1): (!FHE.esint<4>, !FHE.esint<4>) -> (!FHE.esint<4>)

  // CHECK-NEXT: return %[[v0]] : !FHE.esint<4>
  return %0: !FHE.esint<4>
}

// CHECK-LABEL: func.func @to_bool(%arg0: !FHE.eint<1>) -> !FHE.ebool
func.func @to_bool(%arg0: !FHE.eint<1>) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.to_bool"(%arg0) : (!FHE.eint<1>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.to_bool"(%arg0): (!FHE.eint<1>) -> (!FHE.ebool)
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @from_bool(%arg0: !FHE.ebool) -> !FHE.eint<1>
func.func @from_bool(%arg0: !FHE.ebool) -> !FHE.eint<1> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.from_bool"(%arg0) : (!FHE.ebool) -> !FHE.eint<1>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<1>

  %1 = "FHE.from_bool"(%arg0): (!FHE.ebool) -> (!FHE.eint<1>)
  return %1: !FHE.eint<1>
}

// CHECK-LABEL: func.func @gen_gate(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: tensor<4xi64>) -> !FHE.ebool
func.func @gen_gate(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: tensor<4xi64>) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.gen_gate"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, tensor<4xi64>) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.gen_gate"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, tensor<4xi64>) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @mux(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: !FHE.ebool) -> !FHE.ebool
func.func @mux(%arg0: !FHE.ebool, %arg1: !FHE.ebool, %arg2: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.mux"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, !FHE.ebool) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.mux"(%arg0, %arg1, %arg2) : (!FHE.ebool, !FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @and(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @and(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.and"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.and"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @or(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @or(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.or"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.or"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @nand(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @nand(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.nand"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.nand"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @xor(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool
func.func @xor(%arg0: !FHE.ebool, %arg1: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.xor"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.xor"(%arg0, %arg1) : (!FHE.ebool, !FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @not(%arg0: !FHE.ebool) -> !FHE.ebool
func.func @not(%arg0: !FHE.ebool) -> !FHE.ebool {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.not"(%arg0) : (!FHE.ebool) -> !FHE.ebool
  // CHECK-NEXT: return %[[V1]] : !FHE.ebool

  %1 = "FHE.not"(%arg0) : (!FHE.ebool) -> !FHE.ebool
  return %1: !FHE.ebool
}

// CHECK-LABEL: func.func @round(%arg0: !FHE.eint<5>) -> !FHE.eint<3>
func.func @round(%arg0: !FHE.eint<5>) -> !FHE.eint<3> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.round"(%arg0) : (!FHE.eint<5>) -> !FHE.eint<3>
  // CHECK-NEXT: return %[[V1]] : !FHE.eint<3>

  %1 = "FHE.round"(%arg0) : (!FHE.eint<5>) -> !FHE.eint<3>
  return %1: !FHE.eint<3>
}

// CHECK-LABEL: func.func @change_partition_src(%arg0: !FHE.eint<4>) -> !FHE.eint<4>
func.func @change_partition_src(%arg0: !FHE.eint<4>) -> !FHE.eint<4> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.change_partition"(%arg0) {src = #FHE.partition<name "tfhers", lwe_dim 761, glwe_dim 1, poly_size 2048, pbs_base_log 23, pbs_level 1>} : (!FHE.eint<4>) -> !FHE.eint<4>
  %0 = "FHE.change_partition"(%arg0) {src = #FHE.partition<name "tfhers", lwe_dim 761, glwe_dim 1, poly_size 2048, pbs_base_log 23, pbs_level 1>} : (!FHE.eint<4>) -> (!FHE.eint<4>)
  // CHECK-NEXT: return %[[v0]] : !FHE.eint<4>
  return %0: !FHE.eint<4>
}

// CHECK-LABEL: func.func @change_partition_dest(%arg0: !FHE.eint<4>) -> !FHE.eint<4>
func.func @change_partition_dest(%arg0: !FHE.eint<4>) -> !FHE.eint<4> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.change_partition"(%arg0) {dest = #FHE.partition<name "tfhers", lwe_dim 761, glwe_dim 1, poly_size 2048, pbs_base_log 23, pbs_level 1>} : (!FHE.eint<4>) -> !FHE.eint<4>
  %0 = "FHE.change_partition"(%arg0) {dest = #FHE.partition<name "tfhers", lwe_dim 761, glwe_dim 1, poly_size 2048, pbs_base_log 23, pbs_level 1>} : (!FHE.eint<4>) -> (!FHE.eint<4>)
  // CHECK-NEXT: return %[[v0]] : !FHE.eint<4>
  return %0: !FHE.eint<4>
}

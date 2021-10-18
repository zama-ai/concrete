// RUN: zamacompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func @zero() -> !HLFHE.eint<2>
func @zero() -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[RET:.*]] = "HLFHE.zero"() : () -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[RET]] : !HLFHE.eint<2>

  %1 = "HLFHE.zero"() : () -> !HLFHE.eint<2>
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.add_eint_int"(%arg0, %[[V1]]) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<2>

  %0 = constant 1 : i3
  %1 = "HLFHE.add_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i3) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @sub_int_eint(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @sub_int_eint(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.sub_int_eint"(%[[V1]], %arg0) : (i3, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<2>

  %0 = constant 1 : i3
  %1 = "HLFHE.sub_int_eint"(%0, %arg0): (i3, !HLFHE.eint<2>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @mul_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @mul_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i3
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.mul_eint_int"(%arg0, %[[V1]]) : (!HLFHE.eint<2>, i3) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<2>

  %0 = constant 1 : i3
  %1 = "HLFHE.mul_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i3) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @add_eint(%arg0: !HLFHE.eint<2>, %arg1: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @add_eint(%arg0: !HLFHE.eint<2>, %arg1: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.add_eint"(%arg0, %arg1) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<2>

  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<2>, !HLFHE.eint<2>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<4xi64>) -> !HLFHE.eint<2>
func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: tensor<4xi64>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.apply_lookup_table"(%arg0, %arg1) : (!HLFHE.eint<2>, tensor<4xi64>) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<2>

  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, tensor<4xi64>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>, %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
func @dot_eint_int(%arg0: tensor<2x!HLFHE.eint<2>>,
                   %arg1: tensor<2xi3>) -> !HLFHE.eint<2>
{
  // CHECK-NEXT: %[[RET:.*]] = "HLFHE.dot_eint_int"(%arg0, %arg1) : (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>
  %ret = "HLFHE.dot_eint_int"(%arg0, %arg1) :
    (tensor<2x!HLFHE.eint<2>>, tensor<2xi3>) -> !HLFHE.eint<2>

  //CHECK-NEXT: return %[[RET]] : !HLFHE.eint<2>
  return %ret : !HLFHE.eint<2>
}

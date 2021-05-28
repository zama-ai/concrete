// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @add_eint_int(%arg0: !HLFHE.eint<0>) -> !HLFHE.eint<0>
func @add_eint_int(%arg0: !HLFHE.eint<0>) -> !HLFHE.eint<0> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.add_eint_int"(%arg0, %[[V1]]) : (!HLFHE.eint<0>, i32) -> !HLFHE.eint<0>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<0>

  %0 = constant 1 : i32
  %1 = "HLFHE.add_eint_int"(%arg0, %0): (!HLFHE.eint<0>, i32) -> (!HLFHE.eint<0>)
  return %1: !HLFHE.eint<0>
}

// CHECK-LABEL: func @mul_eint_int(%arg0: !HLFHE.eint<0>) -> !HLFHE.eint<0>
func @mul_eint_int(%arg0: !HLFHE.eint<0>) -> !HLFHE.eint<0> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.mul_eint_int"(%arg0, %[[V1]]) : (!HLFHE.eint<0>, i32) -> !HLFHE.eint<0>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<0>

  %0 = constant 1 : i32
  %1 = "HLFHE.mul_eint_int"(%arg0, %0): (!HLFHE.eint<0>, i32) -> (!HLFHE.eint<0>)
  return %1: !HLFHE.eint<0>
}

// CHECK-LABEL: func @add_eint(%arg0: !HLFHE.eint<0>, %arg1: !HLFHE.eint<0>) -> !HLFHE.eint<0>
func @add_eint(%arg0: !HLFHE.eint<0>, %arg1: !HLFHE.eint<0>) -> !HLFHE.eint<0> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.add_eint"(%arg0, %arg1) : (!HLFHE.eint<0>, !HLFHE.eint<0>) -> !HLFHE.eint<0>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<0>

  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<0>, !HLFHE.eint<0>) -> (!HLFHE.eint<0>)
  return %1: !HLFHE.eint<0>
}

// CHECK-LABEL: func @mul_eint(%arg0: !HLFHE.eint<0>, %arg1: !HLFHE.eint<0>) -> !HLFHE.eint<0>
func @mul_eint(%arg0: !HLFHE.eint<0>, %arg1: !HLFHE.eint<0>) -> !HLFHE.eint<0> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.mul_eint"(%arg0, %arg1) : (!HLFHE.eint<0>, !HLFHE.eint<0>) -> !HLFHE.eint<0>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<0>

  %1 = "HLFHE.mul_eint"(%arg0, %arg1): (!HLFHE.eint<0>, !HLFHE.eint<0>) -> (!HLFHE.eint<0>)
  return %1: !HLFHE.eint<0>
}

// CHECK-LABEL: func @neg_eint(%arg0: !HLFHE.eint<0>) -> !HLFHE.eint<0>
func @neg_eint(%arg0: !HLFHE.eint<0>) -> !HLFHE.eint<0> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.neg_eint"(%arg0) : (!HLFHE.eint<0>) -> !HLFHE.eint<0>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<0>

  %1 = "HLFHE.neg_eint"(%arg0): (!HLFHE.eint<0>) -> (!HLFHE.eint<0>)
  return %1: !HLFHE.eint<0>
}

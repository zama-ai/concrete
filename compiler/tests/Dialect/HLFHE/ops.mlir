// RUN: zamacompiler %s  2>&1| FileCheck %s

// CHECK-LABEL: func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @add_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.add_eint_int"(%arg0, %[[V1]]) : (!HLFHE.eint<2>, i32) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<2>

  %0 = constant 1 : i32
  %1 = "HLFHE.add_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i32) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @mul_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @mul_eint_int(%arg0: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = constant 1 : i32
  // CHECK-NEXT: %[[V2:.*]] = "HLFHE.mul_eint_int"(%arg0, %[[V1]]) : (!HLFHE.eint<2>, i32) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V2]] : !HLFHE.eint<2>

  %0 = constant 1 : i32
  %1 = "HLFHE.mul_eint_int"(%arg0, %0): (!HLFHE.eint<2>, i32) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @add_eint(%arg0: !HLFHE.eint<2>, %arg1: !HLFHE.eint<2>) -> !HLFHE.eint<2>
func @add_eint(%arg0: !HLFHE.eint<2>, %arg1: !HLFHE.eint<2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.add_eint"(%arg0, %arg1) : (!HLFHE.eint<2>, !HLFHE.eint<2>) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<2>

  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<2>, !HLFHE.eint<2>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: memref<4xi2>) -> !HLFHE.eint<2>
func @apply_lookup_table(%arg0: !HLFHE.eint<2>, %arg1: memref<4xi2>) -> !HLFHE.eint<2> {
  // CHECK-NEXT: %[[V1:.*]] = "HLFHE.apply_lookup_table"(%arg0, %arg1) : (!HLFHE.eint<2>, memref<4xi2>) -> !HLFHE.eint<2>
  // CHECK-NEXT: return %[[V1]] : !HLFHE.eint<2>

  %1 = "HLFHE.apply_lookup_table"(%arg0, %arg1): (!HLFHE.eint<2>, memref<4xi2>) -> (!HLFHE.eint<2>)
  return %1: !HLFHE.eint<2>
}

// CHECK-LABEL: func @dot_eint_int(%arg0: memref<2x!HLFHE.eint<2>>, %arg1: memref<2xi32>, %arg2: memref<!HLFHE.eint<2>>)
func @dot_eint_int(%arg0: memref<2x!HLFHE.eint<2>>,
          %arg1: memref<2xi32>,
          %arg2: memref<!HLFHE.eint<2>>)
{
  // CHECK-NEXT: "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) : (memref<2x!HLFHE.eint<2>>, memref<2xi32>, memref<!HLFHE.eint<2>>) -> ()
  "HLFHE.dot_eint_int"(%arg0, %arg1, %arg2) :
    (memref<2x!HLFHE.eint<2>>, memref<2xi32>, memref<!HLFHE.eint<2>>) -> ()

  //CHECK-NEXT: return
  return
}

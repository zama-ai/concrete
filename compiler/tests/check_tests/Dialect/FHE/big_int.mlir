// RUN: concretecompiler --action=roundtrip %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @mul_chunked_eint_int(%arg0: !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64>
func.func @mul_chunked_eint_int(%arg0: !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[V2:.*]] = "FHE.mul_eint_int"(%arg0, %[[V1]]) : (!FHE.chunked_eint<64>, i64) -> !FHE.chunked_eint<64>
  // CHECK-NEXT: return %[[V2]] : !FHE.chunked_eint<64>

  %0 = arith.constant 1 : i64
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.chunked_eint<64>, i64) -> (!FHE.chunked_eint<64>)
  return %1: !FHE.chunked_eint<64>
}

// CHECK-LABEL: func.func @add_chunked_eint_int(%arg0: !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64>
func.func @add_chunked_eint_int(%arg0: !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64> {
  // CHECK-NEXT: %[[V1:.*]] = arith.constant 1 : i64
  // CHECK-NEXT: %[[V2:.*]] = "FHE.add_eint_int"(%arg0, %[[V1]]) : (!FHE.chunked_eint<64>, i64) -> !FHE.chunked_eint<64>
  // CHECK-NEXT: return %[[V2]] : !FHE.chunked_eint<64>

  %0 = arith.constant 1 : i64
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.chunked_eint<64>, i64) -> (!FHE.chunked_eint<64>)
  return %1: !FHE.chunked_eint<64>
}

// CHECK-LABEL: func.func @add_chunked_eint(%arg0: !FHE.chunked_eint<64>, %arg1: !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64>
func.func @add_chunked_eint(%arg0: !FHE.chunked_eint<64>, %arg1: !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64> {
  // CHECK-NEXT: %[[V1:.*]] = "FHE.add_eint"(%arg0, %arg1) : (!FHE.chunked_eint<64>, !FHE.chunked_eint<64>) -> !FHE.chunked_eint<64>
  // CHECK-NEXT: return %[[V1]] : !FHE.chunked_eint<64>

  %1 = "FHE.add_eint"(%arg0, %arg1): (!FHE.chunked_eint<64>, !FHE.chunked_eint<64>) -> (!FHE.chunked_eint<64>)
  return %1: !FHE.chunked_eint<64>
}

// TODO: max/min

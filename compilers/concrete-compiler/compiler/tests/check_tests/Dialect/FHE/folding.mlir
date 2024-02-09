// RUN: concretecompiler --action=dump-fhe --optimizer-strategy=V0 --skip-program-info %s 2>&1| FileCheck %s

// CHECK-LABEL: func.func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @add_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: return %arg0 : !FHE.eint<2>

  %0 = arith.constant 0 : i3
  %1 = "FHE.add_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @sub_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @sub_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: return %arg0 : !FHE.eint<2>

  %0 = arith.constant 0 : i3
  %1 = "FHE.sub_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @mul_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @mul_eint_int(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: return %arg0 : !FHE.eint<2>

  %0 = arith.constant 1 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @mul_eint_int_zero(%arg0: !FHE.eint<2>) -> !FHE.eint<2>
func.func @mul_eint_int_zero(%arg0: !FHE.eint<2>) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.zero"()
  // CHECK-NEXT: return %[[v0]] : !FHE.eint<2>

  %0 = arith.constant 0 : i3
  %1 = "FHE.mul_eint_int"(%arg0, %0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @mul_eint_zero_int(%arg0: i3) -> !FHE.eint<2>
func.func @mul_eint_zero_int(%arg0: i3) -> !FHE.eint<2> {
  // CHECK-NEXT: %[[v0:.*]] = "FHE.zero"()
  // CHECK-NEXT: return %[[v0]] : !FHE.eint<2>

  %0 = "FHE.zero"() : () -> !FHE.eint<2>
  %1 = "FHE.mul_eint_int"(%0, %arg0): (!FHE.eint<2>, i3) -> (!FHE.eint<2>)
  return %1: !FHE.eint<2>
}

// CHECK-LABEL: func.func @round(%arg0: !FHE.eint<5>) -> !FHE.eint<5>
func.func @round(%arg0: !FHE.eint<5>) -> !FHE.eint<5> {
  // CHECK-NEXT: return %arg0 : !FHE.eint<5>

  %1 = "FHE.round"(%arg0) : (!FHE.eint<5>) -> !FHE.eint<5>
  return %1: !FHE.eint<5>
}

// CHECK: func.func @to_signed_zero() -> !FHE.esint<7> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero"()
// CHECK-NEXT: return %[[v0]]
func.func @to_signed_zero() -> !FHE.esint<7> {
  %0 = "FHE.zero"() : () -> !FHE.eint<7>
  %1 = "FHE.to_signed"(%0) : (!FHE.eint<7>) -> !FHE.esint<7>
  return %1 : !FHE.esint<7>
}

// CHECK: func.func @to_unsigned_zero() -> !FHE.eint<7> {
// CHECK-NEXT: %[[v0:.*]] = "FHE.zero"()
// CHECK-NEXT: return %[[v0]]
func.func @to_unsigned_zero() -> !FHE.eint<7> {
  %0 = "FHE.zero"() : () -> !FHE.esint<7>
  %1 = "FHE.to_unsigned"(%0) : (!FHE.esint<7>) -> !FHE.eint<7>
  return %1 : !FHE.eint<7>
}

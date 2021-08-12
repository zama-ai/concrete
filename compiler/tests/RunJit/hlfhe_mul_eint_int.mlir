// RUN: zamacompiler %s --run-jit --jit-args 21 --jit-args 2 2>&1| FileCheck %s

// CHECK-LABEL: 42
func @main(%arg0: !HLFHE.eint<7>, %0: i8) -> !HLFHE.eint<7> {
  %1 = "HLFHE.mul_eint_int"(%arg0, %0): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
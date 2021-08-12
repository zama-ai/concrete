// RUN: zamacompiler %s --run-jit --jit-args 2 --jit-args 3 2>&1| FileCheck %s

// CHECK-LABEL: 12
func @main(%arg0: !HLFHE.eint<7>, %arg1: i8) -> !HLFHE.eint<7> {
  %1 = "HLFHE.mul_eint_int"(%arg0, %arg1): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
  %2 = "HLFHE.add_eint"(%1, %1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %2: !HLFHE.eint<7>
}
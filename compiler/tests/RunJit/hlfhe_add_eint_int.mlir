// RUN: zamacompiler %s --run-jit --jit-args 12 --jit-args 30 2>&1| FileCheck %s

// CHECK-LABEL: 42
func @main(%arg0: !HLFHE.eint<7>, %arg1: i8) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint_int"(%arg0, %arg1): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
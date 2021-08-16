// RUN: zamacompiler %s --run-jit --jit-args 100 --jit-args 27 2>&1| FileCheck %s

// CHECK-LABEL: 127
func @main(%arg0: !HLFHE.eint<7>, %arg1: i8) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint_int"(%arg0, %arg1): (!HLFHE.eint<7>, i8) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
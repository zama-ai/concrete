// RUN: zamacompiler %s --run-jit --jit-args 2 --jit-args 3 2>&1| FileCheck %s

// CHECK-LABEL: 5
func @main(%arg0: !HLFHE.eint<7>, %arg1: !HLFHE.eint<7>) -> !HLFHE.eint<7> {
  %1 = "HLFHE.add_eint"(%arg0, %arg1): (!HLFHE.eint<7>, !HLFHE.eint<7>) -> (!HLFHE.eint<7>)
  return %1: !HLFHE.eint<7>
}
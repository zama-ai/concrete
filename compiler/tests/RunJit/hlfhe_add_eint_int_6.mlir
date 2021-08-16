// RUN: zamacompiler %s --run-jit --jit-args 10 --jit-args 54 2>&1| FileCheck %s

// CHECK-LABEL: 64
func @main(%arg0: !HLFHE.eint<6>, %arg1: i7) -> !HLFHE.eint<6> {
  %1 = "HLFHE.add_eint_int"(%arg0, %arg1): (!HLFHE.eint<6>, i7) -> (!HLFHE.eint<6>)
  return %1: !HLFHE.eint<6>
}
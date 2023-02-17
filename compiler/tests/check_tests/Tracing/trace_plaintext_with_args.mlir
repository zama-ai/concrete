// RUN: concretecompiler --action=jit-invoke --jit-args=1 --funcname=main %s 2>&1| FileCheck %s

// CHECK: Test : 00000100
// CHECK-NEXT: 1
func.func @main(%arg0: !FHE.eint<5>) -> !FHE.eint<5> {
  %0 = arith.constant 4 : i8
  "Tracing.trace_plaintext"(%0){msg="Test"}: (i8) -> ()
  return %arg0: !FHE.eint<5>
}

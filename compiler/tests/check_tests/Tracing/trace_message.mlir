// RUN: concretecompiler --action=jit-invoke --jit-args=1 --funcname=main %s 2>&1| FileCheck %s

// CHECK: Arbitrary message
// CHECK-NEXT: 1
func.func @main(%arg0: !FHE.eint<5>) -> !FHE.eint<5> {
  "Tracing.trace_message"(){msg="Arbitrary message"}: () -> ()
  return %arg0: !FHE.eint<5>
}

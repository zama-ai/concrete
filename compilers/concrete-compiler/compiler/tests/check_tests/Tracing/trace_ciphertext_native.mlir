// RUN: concretecompiler --action=jit-invoke --jit-args=1 --funcname=main %s 2>&1| FileCheck %s

// CHECK: :  [[BODY:[01]{64}]]
// CHECK-NEXT: 1
func.func @main(%arg0: !FHE.eint<5>) -> !FHE.eint<5> {
  "Tracing.trace_ciphertext"(%arg0): (!FHE.eint<5>) -> ()
  return %arg0: !FHE.eint<5>
}

// RUN: concretecompiler --force-encoding=crt --action=jit-invoke --jit-args=1 --funcname=main %s 2>&1| FileCheck %s

// CHECK: Test : [[BODY01:[01]{3}]] [[BODY02:[01]{61}]]
// CHECK-NEXT: Test : [[BODY11:[01]{3}]] [[BODY12:[01]{61}]]
// CHECK-NEXT: Test : [[BODY21:[01]{3}]] [[BODY22:[01]{61}]]
// CHECK-NEXT: 1
func.func @main(%arg0: !FHE.eint<5>) -> !FHE.eint<5> {
  "Tracing.trace_ciphertext"(%arg0){msg="Test", nmsb=3:i32}: (!FHE.eint<5>) -> ()
  return %arg0: !FHE.eint<5>
}

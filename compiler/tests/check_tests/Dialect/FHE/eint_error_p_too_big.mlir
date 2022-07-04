// RUN: not concretecompiler --action=dump-llvm-ir %s  2>&1| FileCheck %s

// CHECK-LABEL: Could not determine V0 parameters
func.func @test(%arg0: !FHE.eint<9>) {
  return
}

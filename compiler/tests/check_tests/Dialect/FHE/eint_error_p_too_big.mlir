// RUN: not concretecompiler --action=dump-llvm-ir %s  2>&1| FileCheck %s

// CHECK-LABEL: Could not determine V0 parameters
func @test(%arg0: !FHE.eint<9>,  %arg1: tensor<512xi64>) {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<9>,  tensor<512xi64>) -> (!FHE.eint<9>)
  return
}

// RUN: not concretecompiler --action=dump-llvm-ir %s  2>&1| FileCheck %s

// CHECK-LABEL: NoParametersFound
func.func @test(%arg0: !FHE.eint<17>,  %arg1: tensor<131072xi64>) -> !FHE.eint<17> {
  %1 = "FHE.apply_lookup_table"(%arg0, %arg1): (!FHE.eint<17>,  tensor<131072xi64>) -> (!FHE.eint<17>)
  return %1 : !FHE.eint<17>
}

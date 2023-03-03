// RUN: concretecompiler %s --action=dump-llvm-dialect --parallelize 2>&1| FileCheck %s

// Check that at some point the compilation pipeline generates a parallel region
// CHECK: omp.parallel
func.func @apply_lookup_table(%arg0: tensor<2x3x4x!FHE.eint<2>>) -> tensor<2x3x4x!FHE.eint<2>> {
  %arg1 = arith.constant dense<"0x0000000000000000000000000000000100000000000000020000000000000003"> : tensor<4xi64>
  %1 = "FHELinalg.apply_lookup_table"(%arg0, %arg1): (tensor<2x3x4x!FHE.eint<2>>, tensor<4xi64>) -> (tensor<2x3x4x!FHE.eint<2>>)
  return %1: tensor<2x3x4x!FHE.eint<2>>
}


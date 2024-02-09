// RUN: concretecompiler --action=dump-llvm-ir --optimizer-strategy=V0 --skip-program-info %s
// Just ensure that compile
// https://github.com/zama-ai/concrete-compiler-internal/issues/785
func.func @main(%arg0: !FHE.eint<5>, %cst: tensor<32xi64>) -> tensor<1x!FHE.eint<5>> {
  %1 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.eint<5>
  %6 = tensor.from_elements %1 : tensor<1x!FHE.eint<5>>   // ERROR HERE line 4
  return %6 : tensor<1x!FHE.eint<5>>
}

// Ensures that tensors of multiple elements can be constructed as well.
func.func @main2(%arg0: !FHE.eint<5>, %cst: tensor<32xi64>) -> tensor<2x!FHE.eint<5>> {
  %1 = "FHE.apply_lookup_table"(%arg0, %cst) : (!FHE.eint<5>, tensor<32xi64>) -> !FHE.eint<5>
  %6 = tensor.from_elements %1, %arg0 : tensor<2x!FHE.eint<5>>   // ERROR HERE line 4
  return %6 : tensor<2x!FHE.eint<5>>
}
